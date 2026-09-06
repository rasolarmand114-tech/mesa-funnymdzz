#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>

#define KBASE_IOCTL_TYPE 0x80

struct kbase_ioctl_version_check { uint16_t major; uint16_t minor; };
#define KBASE_IOCTL_VERSION_CHECK _IOWR(KBASE_IOCTL_TYPE, 0, struct kbase_ioctl_version_check)
struct kbase_ioctl_set_flags { uint32_t create_flags; };
#define KBASE_IOCTL_SET_FLAGS _IOW(KBASE_IOCTL_TYPE, 1, struct kbase_ioctl_set_flags)
struct kbase_ioctl_job_submit { uint64_t addr; uint32_t nr_atoms; uint32_t stride; };
#define KBASE_IOCTL_JOB_SUBMIT _IOW(KBASE_IOCTL_TYPE, 2, struct kbase_ioctl_job_submit)
#define BASE_MEM_PROT_CPU_RD (1ull << 0)
#define BASE_MEM_PROT_CPU_WR (1ull << 1)
#define BASE_MEM_PROT_GPU_RD (1ull << 2)
#define BASE_MEM_PROT_GPU_WR (1ull << 3)
union kbase_ioctl_mem_alloc {
   struct { uint64_t va_pages; uint64_t commit_pages; uint64_t extension; uint64_t flags; } in;
   struct { uint64_t flags; uint64_t gpu_va; } out;
};
#define KBASE_IOCTL_MEM_ALLOC _IOWR(KBASE_IOCTL_TYPE, 5, union kbase_ioctl_mem_alloc)
#define BASE_MEM_MAP_TRACKING_HANDLE (3ull << 12)

static int fd;
static uint64_t atom_va;   /* VA (GPU==CPU, SAME_VA) */

static int submit(uint64_t addr, uint32_t nr, uint32_t stride) {
   struct kbase_ioctl_job_submit s = { .addr = addr, .nr_atoms = nr, .stride = stride };
   errno = 0;
   long r = ioctl(fd, KBASE_IOCTL_JOB_SUBMIT, &s);
   return (r == 0) ? 0 : -errno;
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

   union kbase_ioctl_mem_alloc alloc = { 0 };
   alloc.in.va_pages = 4;
   alloc.in.commit_pages = 4;
   alloc.in.flags = BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_CPU_WR |
                    BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_WR;
   if (ioctl(fd, KBASE_IOCTL_MEM_ALLOC, &alloc)) { printf("mem_alloc: %s\n", strerror(errno)); return 1; }
   printf("MEM_ALLOC cookie=0x%llx flags=0x%llx\n",
          (unsigned long long)alloc.out.gpu_va, (unsigned long long)alloc.out.flags);

   void *cpu = mmap(NULL, 4096 * 4, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                    (off_t)alloc.out.gpu_va);
   if (cpu == MAP_FAILED) { printf("mmap VA: %s\n", strerror(errno)); return 1; }
   memset(cpu, 0, 4096 * 4);
   atom_va = (uint64_t)(uintptr_t)cpu;
   printf("VA (GPU==CPU) = 0x%llx\n", (unsigned long long)atom_va);

   uint8_t *at = cpu;
   at[sizeof(uint64_t) * 0 + 0] = 0;                     /* jc = 0 */
   at[40] = 1;                                           /* atom_number = 1 */

   printf("\n-- bissecção de stride (atom zerado em memória GPU) --\n");
   static const uint32_t strides[] = { 0, 16, 40, 44, 48, 52, 56, 60, 64 };
   for (unsigned i = 0; i < sizeof(strides) / sizeof(strides[0]); i++) {
      int e = submit(atom_va, 1, strides[i]);
      printf("stride=%-3u -> errno=%d (%s)  %s\n",
             strides[i], -e, e == 0 ? "OK" : strerror(-e),
             (strides[i] == 0) ? "<- esperava EINVAL (controle)" :
             (e == 0) ? "<- ACEITO!" : "");
   }

   printf("\n-- outros controles (stride=48) --\n");
   printf("addr=0          -> errno=%d\n", -submit(0, 1, 48));
   printf("addr=va+0x4     -> errno=%d\n", -submit(atom_va + 4, 1, 48));
   printf("addr=va+0x1000  -> errno=%d\n", -submit(atom_va + 0x1000, 1, 48));
   printf("nr_atoms=0      -> errno=%d\n", -submit(atom_va, 0, 48));
   printf("nr_atoms=2      -> errno=%d\n", -submit(atom_va, 2, 48));
   printf("stride=0        -> errno=%d\n", -submit(atom_va, 1, 0));

   printf("\n-- tentativa de completar (stride=56, laço 56&pad=0) --\n");
   {
      at[40] = 1;
      int e = submit(atom_va, 1, 56);
      printf("stride=56       -> %s\n", e == 0 ? "JOB_SUBMIT OK" : strerror(errno));
      if (e == 0) {
         struct pollfd pfd = { .fd = fd, .events = POLLIN };
         int p = poll(&pfd, 1, 3000);
         printf("poll=%d revents=0x%x\n", p, pfd.revents);
         if (p > 0) {
            uint8_t ev[64]; memset(ev, 0, sizeof(ev));
            ssize_t n = read(fd, ev, sizeof(ev));
            printf("event %zd bytes: buf[0]=0x%x buf[4]=0x%x\n", n, ev[0], ev[4]);
         } else printf("sem evento (timeout)\n");
      }
   }
   close(fd);
   return 0;
}
