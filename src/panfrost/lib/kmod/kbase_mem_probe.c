//*kbase_mem_probe.c 
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>

#define KBASE_IOCTL_TYPE 0x80

struct kbase_ioctl_version_check { uint16_t major; uint16_t minor; };
#define KBASE_IOCTL_VERSION_CHECK _IOWR(KBASE_IOCTL_TYPE, 0, struct kbase_ioctl_version_check)
struct kbase_ioctl_set_flags { uint32_t create_flags; };
#define KBASE_IOCTL_SET_FLAGS _IOW(KBASE_IOCTL_TYPE, 1, struct kbase_ioctl_set_flags)
#define BASE_MEM_PROT_CPU_RD (1ull << 0)
#define BASE_MEM_PROT_CPU_WR (1ull << 1)
#define BASE_MEM_PROT_GPU_RD (1ull << 2)
#define BASE_MEM_PROT_GPU_WR (1ull << 3)
#define BASE_MEM_PROT_GPU_EX (1ull << 4)
#define BASE_MEM_COHERENT_SYSTEM (1ull << 10)
union kbase_ioctl_mem_alloc {
   struct { uint64_t va_pages; uint64_t commit_pages; uint64_t extension; uint64_t flags; } in;
   struct { uint64_t flags; uint64_t gpu_va; } out;
};
#define KBASE_IOCTL_MEM_ALLOC _IOWR(KBASE_IOCTL_TYPE, 5, union kbase_ioctl_mem_alloc)
struct kbase_ioctl_mem_jit_init { uint64_t va_pages; uint8_t max_allocations; uint8_t trim_level; uint8_t group_id; uint8_t padding[5]; uint64_t phys_pages; };
#define KBASE_IOCTL_MEM_JIT_INIT _IOW(KBASE_IOCTL_TYPE, 14, struct kbase_ioctl_mem_jit_init)
struct kbase_ioctl_mem_exec_init { uint64_t va_pages; };
#define KBASE_IOCTL_MEM_EXEC_INIT _IOW(KBASE_IOCTL_TYPE, 38, struct kbase_ioctl_mem_exec_init)
struct kbase_ioctl_mem_query { /* in: gpu_addr, query ; out: value */
   union { struct { uint64_t gpu_addr; uint64_t query; } in; uint64_t out; };
};
#define KBASE_IOCTL_MEM_QUERY _IOWR(KBASE_IOCTL_TYPE, 6, struct kbase_ioctl_mem_query)
#define KBASE_MEM_QUERY_FLAGS 3
#define BASE_MEM_MAP_TRACKING_HANDLE (3ull << 12)

static int fd;

static int do_alloc(int npages, uint64_t flags, const char *tag) {
   union kbase_ioctl_mem_alloc a = { 0 };
   a.in.va_pages = npages;
   a.in.commit_pages = npages;
   a.in.flags = flags;
   errno = 0;
   if (ioctl(fd, KBASE_IOCTL_MEM_ALLOC, &a)) {
      printf("  %-28s -> MEM_ALLOC FAIL: %s\n", tag, strerror(errno));
      return -1;
   }
   void *cpu = mmap(NULL, 4096 * (size_t)npages, PROT_READ | PROT_WRITE, MAP_SHARED,
                    fd, (off_t)a.out.gpu_va);
   if (cpu == MAP_FAILED) {
      printf("  %-28s -> alloc ok cookie=0x%llx flags=0x%llx mas mmap FAIL: %s\n",
             tag, (unsigned long long)a.out.gpu_va,
             (unsigned long long)a.out.flags, strerror(errno));
      return -1;
   }
   printf("  %-28s -> ok cookie=0x%llx flags=0x%llx VA=0x%llx\n",
          tag, (unsigned long long)a.out.gpu_va,
          (unsigned long long)a.out.flags,
          (unsigned long long)(uintptr_t)cpu);
   return 0;
}

int main(void) {
   fd = open("/dev/mali0", O_RDWR);
   if (fd < 0) { printf("open: %s\n", strerror(errno)); return 1; }
   struct kbase_ioctl_version_check ver = {0, 0};
   if (ioctl(fd, KBASE_IOCTL_VERSION_CHECK, &ver)) { printf("version: %s\n", strerror(errno)); return 1; }
   printf("VERSION_CHECK %u.%u\n", ver.major, ver.minor);
   struct kbase_ioctl_set_flags sf = { .create_flags = 0 };
   if (ioctl(fd, KBASE_IOCTL_SET_FLAGS, &sf)) { printf("set_flags: %s\n", strerror(errno)); return 1; }
   if (mmap(NULL, 4096, PROT_NONE, MAP_SHARED, fd, BASE_MEM_MAP_TRACKING_HANDLE) == MAP_FAILED)
      { printf("tracking: %s\n", strerror(errno)); return 1; }
   printf("handshake ok\n");

   printf("\n-- MEM_EXEC_INIT (zona de memória executável) --\n");
   struct kbase_ioctl_mem_exec_init ex = { .va_pages = 0x40 };   /* 256KB */
   errno = 0;
   if (ioctl(fd, KBASE_IOCTL_MEM_EXEC_INIT, &ex))
      printf("  MEM_EXEC_INIT(256K) FAIL: %s\n", strerror(errno));
   else
      printf("  MEM_EXEC_INIT(256K) ok\n");
   ex.va_pages = 0x4000;
   errno = 0;
   if (ioctl(fd, KBASE_IOCTL_MEM_EXEC_INIT, &ex))
      printf("  MEM_EXEC_INIT(64MB)  FAIL: %s\n", strerror(errno));
   else
      printf("  MEM_EXEC_INIT(64MB)  ok\n");

   printf("\n-- MEM_JIT_INIT --\n");
   struct kbase_ioctl_mem_jit_init ji = { 0 };
   ji.va_pages = 0x100;
   ji.max_allocations = 8;
   errno = 0;
   if (ioctl(fd, KBASE_IOCTL_MEM_JIT_INIT, &ji))
      printf("  MEM_JIT_INIT         FAIL: %s\n", strerror(errno));
   else
      printf("  MEM_JIT_INIT         ok\n");

   printf("\n-- MEM_ALLOC com diferentes flag sets --\n");
   do_alloc(1, BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_CPU_WR | BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_WR, "RW padrão");
   do_alloc(1, BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_CPU_WR | BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_WR | BASE_MEM_PROT_GPU_EX, "RW + GPU_EX");
   do_alloc(1, BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_CPU_WR | BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_WR | BASE_MEM_COHERENT_SYSTEM, "RW + Coherent SYSTEM");
   do_alloc(1, BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_CPU_WR | BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_WR | BASE_MEM_PROT_GPU_EX | BASE_MEM_COHERENT_SYSTEM, "RW + EX + Coherent");

   close(fd);
   return 0;
}
