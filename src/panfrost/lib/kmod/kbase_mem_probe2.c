/*kbase_mem_probe2.c
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
#define BASE_MEM_PROT_CPU_RD (1ull << 0)
#define BASE_MEM_PROT_CPU_WR (1ull << 1)
#define BASE_MEM_PROT_GPU_RD (1ull << 2)
#define BASE_MEM_PROT_GPU_WR (1ull << 3)
#define BASE_MEM_PROT_GPU_EX (1ull << 4)
#define BASE_MEM_COHERENT_LOCAL (1ull << 11)
#define BASE_MEM_COHERENT_SYSTEM (1ull << 10)
struct kbase_ioctl_version_check { uint16_t major; uint16_t minor; };
#define KBASE_IOCTL_VERSION_CHECK _IOWR(KBASE_IOCTL_TYPE, 0, struct kbase_ioctl_version_check)
struct kbase_ioctl_set_flags { uint32_t create_flags; };
#define KBASE_IOCTL_SET_FLAGS _IOW(KBASE_IOCTL_TYPE, 1, struct kbase_ioctl_set_flags)
union kbase_ioctl_mem_alloc {
   struct { uint64_t va_pages; uint64_t commit_pages; uint64_t extension; uint64_t flags; } in;
   struct { uint64_t flags; uint64_t gpu_va; } out;
};
#define KBASE_IOCTL_MEM_ALLOC _IOWR(KBASE_IOCTL_TYPE, 5, union kbase_ioctl_mem_alloc)
struct kbase_ioctl_mem_exec_init { uint64_t va_pages; };
#define KBASE_IOCTL_MEM_EXEC_INIT _IOW(KBASE_IOCTL_TYPE, 38, struct kbase_ioctl_mem_exec_init)
#define BASE_MEM_MAP_TRACKING_HANDLE (3ull << 12)

static int fresh(void) {
   int fd = open("/dev/mali0", O_RDWR);
   if (fd < 0) { printf("  open FAIL: %s\n", strerror(errno)); return -1; }
   struct kbase_ioctl_version_check v = {0, 0};
   if (ioctl(fd, KBASE_IOCTL_VERSION_CHECK, &v)) { printf("  version: %s\n", strerror(errno)); return -1; }
   struct kbase_ioctl_set_flags sf = { .create_flags = 0 };
   if (ioctl(fd, KBASE_IOCTL_SET_FLAGS, &sf)) { printf("  set_flags: %s\n", strerror(errno)); return -1; }
   if (mmap(NULL, 4096, PROT_NONE, MAP_SHARED, fd, BASE_MEM_MAP_TRACKING_HANDLE) == MAP_FAILED)
      { printf("  tracking: %s\n", strerror(errno)); return -1; }
   return fd;
}

/* contexto limpo, testa uma combinação exec. */
static void exec_case(const char *tag, uint64_t exec_pages, uint64_t alloc_flags) {
   printf("\n-- %s --\n", tag);
   int fd = fresh();
   if (fd < 0) return;

   struct kbase_ioctl_mem_exec_init ex = { .va_pages = exec_pages };
   errno = 0;
   if (ioctl(fd, KBASE_IOCTL_MEM_EXEC_INIT, &ex)) {
      printf("  MEM_EXEC_INIT(%llu pags) FAIL: %s\n",
             (unsigned long long)exec_pages, strerror(errno));
      close(fd); return;
   }
   printf("  MEM_EXEC_INIT(%llu pags = %llu MB) ok\n",
          (unsigned long long)exec_pages,
          (unsigned long long)(exec_pages >> 8));

   union kbase_ioctl_mem_alloc a = { 0 };
   a.in.va_pages = 1;
   a.in.commit_pages = 1;
   a.in.flags = alloc_flags;
   errno = 0;
   if (ioctl(fd, KBASE_IOCTL_MEM_ALLOC, &a)) {
      printf("  MEM_ALLOC FAIL: %s\n", strerror(errno));
      close(fd); return;
   }
   printf("  MEM_ALLOC ok: cookie/VA out=0x%llx flags out=0x%llx\n",
          (unsigned long long)a.out.gpu_va, (unsigned long long)a.out.flags);
   printf("  %s\n", (a.out.flags & (1ull << 13)) ? "  !! SAME_VA set (inesperado para EX)" : "  SEM SAME_VA (esperado p/ EXEC_VA)");

   void *cpu = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)a.out.gpu_va);
   if (cpu == MAP_FAILED) {
      printf("  mmap FAIL: %s (o VA de ex é real; se mmap pega, VA==out.gpu_va)\n", strerror(errno));
      close(fd); return;
   }
   printf("  mmap ok: cpu=0x%p\n", cpu);
   /* GPU VA REAL para executável é out.gpu_va; CPU ptr pode diferir. */
   printf("  -> GPU VA do shader = 0x%llx (usar isso em job jc/descritor)\n",
          (unsigned long long)((a.out.flags & (1ull << 13)) ? (uintptr_t)cpu : a.out.gpu_va));
   close(fd);
}

int main(void) {
   uint64_t mesa_non_w = BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_CPU_WR |
                         BASE_MEM_PROT_GPU_RD | BASE_MEM_COHERENT_LOCAL |
                         BASE_MEM_PROT_GPU_EX;   /* COMO O MESA: GPU_EX SEM GPU_WR (W^X), sem SAME_VA */

   exec_case("T1: exec-zone 1GB + alloc EX (mesa, sem GPU_WR)", 0x40000, mesa_non_w);
   exec_case("T2: exec-zone 1GB + alloc EX|GPU_WR (W^X) -> espera ENOMEM", 0x40000,
             mesa_non_w | BASE_MEM_PROT_GPU_WR);
   exec_case("T3: exec-zone pequena 256KB + alloc EX (mesa, sem GPU_WR)", 0x40, mesa_non_w);
   exec_case("T4: exec-zone 4GB (0x100000, como o mesa) + alloc EX", 0x100000, mesa_non_w);
   return 0;
}
