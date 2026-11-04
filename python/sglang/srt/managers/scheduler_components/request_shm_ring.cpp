// Shared-memory request ring fed by a native zmq intake thread.
//
// The leader rank's writer thread pulls tokenizer frames off the scheduler
// input socket without the GIL and appends them to a single-producer ring in
// POSIX shared memory. Every rank of the group (leader included) maps the ring
// and reads the same records, so request payloads never cross a gloo
// broadcast; the ranks only agree on the last sequence number to consume.
//
// Layout: Header, then `capacity` bytes of 64-byte aligned records
// [RecordHdr][payload]. Offsets are monotonic byte counters; the physical
// position is offset % capacity and a record never spans the end (a wrap
// marker skips the tail). A frame larger than capacity / 4 is spilled to its
// own segment; the last reader to release it unlinks it.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zmq.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

constexpr uint64_t kMagic = 0x474e495252474c53ULL;  // "SLGRRING"
constexpr uint32_t kMaxReaders = 64;
constexpr uint64_t kAlign = 64;
constexpr uint64_t kWrapMarker = ~0ULL;
constexpr uint64_t kSpillFlag = 1ULL << 63;

struct alignas(64) Header {
  uint64_t magic;
  uint64_t capacity;
  uint32_t num_readers;
  uint32_t reserved;
  alignas(64) std::atomic<uint64_t> write_off;
  alignas(64) std::atomic<uint64_t> published_seq;
  alignas(64) std::atomic<uint64_t> reader_off[kMaxReaders];
  alignas(64) std::atomic<uint64_t> stat_msgs;
  std::atomic<uint64_t> stat_bytes;
  std::atomic<uint64_t> stat_spills;
  std::atomic<uint64_t> stat_full_wait_ns;
  std::atomic<uint64_t> stat_dropped_frames;
};

struct RecordHdr {
  uint64_t seq;
  uint64_t len;  // payload bytes; kWrapMarker; or kSpillFlag | spilled bytes
};

struct SpillHdr {
  std::atomic<uint32_t> refs;
  uint32_t reserved;
};

inline uint64_t align_up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string spill_name(const std::string& base, uint64_t seq) {
  return base + ".s" + std::to_string(seq);
}

void* map_segment(const std::string& name, size_t size, bool create, bool populate) {
  int flags = create ? (O_CREAT | O_EXCL | O_RDWR) : O_RDWR;
  int fd = shm_open(name.c_str(), flags, 0600);
  if (fd < 0) throw std::runtime_error("shm_open failed for " + name + ": " + strerror(errno));
  if (create && ftruncate(fd, static_cast<off_t>(size)) != 0) {
    close(fd);
    shm_unlink(name.c_str());
    throw std::runtime_error("ftruncate failed for " + name);
  }
  int mflags = MAP_SHARED | (populate ? MAP_POPULATE : 0);
  void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, mflags, fd, 0);
  close(fd);
  if (addr == MAP_FAILED) throw std::runtime_error("mmap failed for " + name);
  return addr;
}

class RequestShmRing {
 public:
  // Leader: create the ring and start the intake thread on `endpoint`.
  RequestShmRing(const std::string& name, uint64_t capacity, uint32_t num_readers,
                 const std::string& endpoint, int64_t rcvbuf)
      : name_(name), reader_index_(0) {
    if (num_readers == 0 || num_readers > kMaxReaders) throw std::invalid_argument("bad num_readers");
    capacity = align_up(capacity);
    total_size_ = sizeof(Header) + capacity;
    base_ = map_segment(name_, total_size_, /*create=*/true, /*populate=*/true);
    hdr_ = new (base_) Header();
    hdr_->magic = kMagic;
    hdr_->capacity = capacity;
    hdr_->num_readers = num_readers;
    data_ = static_cast<uint8_t*>(base_) + sizeof(Header);
    start_writer(endpoint, rcvbuf);
  }

  // Follower: attach to an existing ring as reader `reader_index`.
  RequestShmRing(const std::string& name, uint32_t reader_index)
      : name_(name), reader_index_(reader_index) {
    int fd = shm_open(name_.c_str(), O_RDWR, 0600);
    if (fd < 0) throw std::runtime_error("shm_open (attach) failed for " + name_);
    struct stat st;
    fstat(fd, &st);
    close(fd);
    total_size_ = static_cast<size_t>(st.st_size);
    base_ = map_segment(name_, total_size_, /*create=*/false, /*populate=*/false);
    hdr_ = static_cast<Header*>(base_);
    if (hdr_->magic != kMagic) throw std::runtime_error("request ring magic mismatch");
    if (reader_index_ >= hdr_->num_readers) throw std::invalid_argument("reader index out of range");
    data_ = static_cast<uint8_t*>(base_) + sizeof(Header);
  }

  ~RequestShmRing() { shutdown(); }

  void shutdown() {
    if (writer_.joinable()) {
      stop_.store(true);
      writer_.join();
    }
    release_spills();
    if (base_ != nullptr) {
      munmap(base_, total_size_);
      base_ = nullptr;
    }
  }

  // The name is no longer needed once every reader has mapped the ring.
  void unlink_name() { shm_unlink(name_.c_str()); }

  uint64_t published_seq() const { return hdr_->published_seq.load(std::memory_order_acquire); }

  // Seq of the `max_count`-th unread record (0 = unlimited), capped by what is
  // published. Leader-only: decides how much the group consumes this round.
  uint64_t scan_limit(uint64_t max_count) const {
    uint64_t upto = published_seq();
    if (max_count == 0 || upto <= consumed_seq_) return upto;
    return std::min(upto, consumed_seq_ + max_count);
  }

  // Views of every record after this reader's position up to seq `upto`.
  // Valid until release(); the writer cannot reuse the space before that.
  py::list read(uint64_t upto) {
    py::list out;
    if (upto <= consumed_seq_) return out;
    if (upto > published_seq()) throw std::runtime_error("read beyond published sequence");
    uint64_t off = hdr_->reader_off[reader_index_].load(std::memory_order_relaxed);
    const uint64_t cap = hdr_->capacity;
    uint64_t seq = consumed_seq_;
    while (seq < upto) {
      uint64_t pos = off % cap;
      auto* rec = reinterpret_cast<RecordHdr*>(data_ + pos);
      if (rec->len == kWrapMarker) {
        off += cap - pos;
        continue;
      }
      if (rec->seq != seq + 1) throw std::runtime_error("request ring sequence gap");
      seq = rec->seq;
      if (rec->len & kSpillFlag) {
        uint64_t len = rec->len & ~kSpillFlag;
        size_t size = sizeof(SpillHdr) + len;
        void* addr = map_segment(spill_name(name_, seq), size, false, false);
        spills_.push_back({addr, size, seq});
        out.append(py::memoryview::from_memory(
            static_cast<uint8_t*>(addr) + sizeof(SpillHdr), static_cast<ssize_t>(len), true));
        off += align_up(sizeof(RecordHdr));
      } else {
        out.append(py::memoryview::from_memory(data_ + pos + sizeof(RecordHdr),
                                               static_cast<ssize_t>(rec->len), true));
        off += align_up(sizeof(RecordHdr) + rec->len);
      }
    }
    pending_off_ = off;
    pending_seq_ = seq;
    return out;
  }

  // Give the space of the last read() back to the writer.
  void release() {
    if (pending_seq_ <= consumed_seq_) return;
    release_spills();
    consumed_seq_ = pending_seq_;
    hdr_->reader_off[reader_index_].store(pending_off_, std::memory_order_release);
  }

  py::dict stats() const {
    py::dict d;
    d["msgs"] = hdr_->stat_msgs.load();
    d["bytes"] = hdr_->stat_bytes.load();
    d["spills"] = hdr_->stat_spills.load();
    d["full_wait_ms"] = hdr_->stat_full_wait_ns.load() / 1e6;
    d["dropped_frames"] = hdr_->stat_dropped_frames.load();
    d["published_seq"] = published_seq();
    return d;
  }

 private:
  struct Spill {
    void* addr;
    size_t size;
    uint64_t seq;
  };

  void release_spills() {
    for (auto& s : spills_) {
      auto* sh = static_cast<SpillHdr*>(s.addr);
      bool last = sh->refs.fetch_sub(1, std::memory_order_acq_rel) == 1;
      munmap(s.addr, s.size);
      if (last) shm_unlink(spill_name(name_, s.seq).c_str());
    }
    spills_.clear();
  }

  void start_writer(const std::string& endpoint, int64_t rcvbuf) {
    ctx_ = zmq_ctx_new();
    sock_ = zmq_socket(ctx_, ZMQ_PULL);
    int hwm = 0;
    zmq_setsockopt(sock_, ZMQ_RCVHWM, &hwm, sizeof(hwm));
    if (rcvbuf > 0) {
      int buf = static_cast<int>(rcvbuf);
      zmq_setsockopt(sock_, ZMQ_RCVBUF, &buf, sizeof(buf));
    }
    int timeout_ms = 50;  // bounds shutdown latency only
    zmq_setsockopt(sock_, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
    if (zmq_connect(sock_, endpoint.c_str()) != 0) {
      throw std::runtime_error(std::string("zmq_connect failed: ") + zmq_strerror(zmq_errno()));
    }
    writer_ = std::thread([this] { writer_loop(); });
  }

  uint64_t min_reader_off() const {
    uint64_t m = ~0ULL;
    for (uint32_t i = 0; i < hdr_->num_readers; ++i) {
      m = std::min(m, hdr_->reader_off[i].load(std::memory_order_acquire));
    }
    return m;
  }

  bool wait_space(uint64_t end) {
    const uint64_t cap = hdr_->capacity;
    if (end - min_reader_off() <= cap) return true;
    uint64_t t0 = now_ns();
    while (end - min_reader_off() > cap) {
      if (stop_.load()) return false;
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    hdr_->stat_full_wait_ns.fetch_add(now_ns() - t0);
    return true;
  }

  void append(const uint8_t* payload, uint64_t len) {
    const uint64_t cap = hdr_->capacity;
    const uint64_t seq = next_seq_;
    const bool spill = len > cap / 4;
    uint64_t need = spill ? align_up(sizeof(RecordHdr)) : align_up(sizeof(RecordHdr) + len);
    if (spill) {
      size_t size = sizeof(SpillHdr) + len;
      void* addr = map_segment(spill_name(name_, seq), size, true, false);
      auto* sh = new (addr) SpillHdr();
      sh->refs.store(hdr_->num_readers);
      std::memcpy(static_cast<uint8_t*>(addr) + sizeof(SpillHdr), payload, len);
      munmap(addr, size);
      hdr_->stat_spills.fetch_add(1);
    }
    uint64_t w = hdr_->write_off.load(std::memory_order_relaxed);
    uint64_t pos = w % cap;
    if (pos + need > cap) {
      if (!wait_space(w + (cap - pos) + need)) return;
      reinterpret_cast<RecordHdr*>(data_ + pos)->len = kWrapMarker;
      w += cap - pos;
      pos = 0;
    } else if (!wait_space(w + need)) {
      return;
    }
    auto* rec = reinterpret_cast<RecordHdr*>(data_ + pos);
    rec->seq = seq;
    if (spill) {
      rec->len = kSpillFlag | len;
    } else {
      rec->len = len;
      std::memcpy(data_ + pos + sizeof(RecordHdr), payload, len);
    }
    w += need;
    hdr_->write_off.store(w, std::memory_order_relaxed);
    next_seq_ = seq + 1;
    hdr_->stat_msgs.fetch_add(1, std::memory_order_relaxed);
    hdr_->stat_bytes.fetch_add(len, std::memory_order_relaxed);
    // Publishes the record (and any wrap marker) to readers.
    hdr_->published_seq.store(seq, std::memory_order_release);
  }

  void writer_loop() {
    zmq_msg_t msg;
    while (!stop_.load()) {
      zmq_msg_init(&msg);
      int rc = zmq_msg_recv(&msg, sock_, 0);
      if (rc < 0) {
        zmq_msg_close(&msg);
        continue;  // EAGAIN on timeout, EINTR
      }
      bool more = zmq_msg_more(&msg);
      append(static_cast<const uint8_t*>(zmq_msg_data(&msg)), zmq_msg_size(&msg));
      zmq_msg_close(&msg);
      while (more) {  // sglang sends single-frame messages; never expected
        zmq_msg_init(&msg);
        zmq_msg_recv(&msg, sock_, 0);
        more = zmq_msg_more(&msg);
        zmq_msg_close(&msg);
        hdr_->stat_dropped_frames.fetch_add(1);
      }
    }
    zmq_close(sock_);
    zmq_ctx_term(ctx_);
  }

  std::string name_;
  uint32_t reader_index_;
  size_t total_size_ = 0;
  void* base_ = nullptr;
  Header* hdr_ = nullptr;
  uint8_t* data_ = nullptr;
  uint64_t consumed_seq_ = 0;
  uint64_t pending_seq_ = 0;
  uint64_t pending_off_ = 0;
  std::vector<Spill> spills_;
  // writer state
  uint64_t next_seq_ = 1;
  void* ctx_ = nullptr;
  void* sock_ = nullptr;
  std::atomic<bool> stop_{false};
  std::thread writer_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<RequestShmRing>(m, "RequestShmRing")
      .def(py::init<const std::string&, uint64_t, uint32_t, const std::string&, int64_t>(),
           py::arg("name"), py::arg("capacity"), py::arg("num_readers"), py::arg("endpoint"),
           py::arg("rcvbuf"))
      .def(py::init<const std::string&, uint32_t>(), py::arg("name"), py::arg("reader_index"))
      .def("published_seq", &RequestShmRing::published_seq)
      .def("scan_limit", &RequestShmRing::scan_limit, py::arg("max_count"))
      .def("read", &RequestShmRing::read, py::arg("upto"))
      .def("release", &RequestShmRing::release)
      .def("unlink_name", &RequestShmRing::unlink_name)
      .def("stats", &RequestShmRing::stats)
      .def("shutdown", &RequestShmRing::shutdown, py::call_guard<py::gil_scoped_release>());
}
