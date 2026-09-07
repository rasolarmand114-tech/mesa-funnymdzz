#ifndef PANVK_KBASE_FD_H
#define PANVK_KBASE_FD_H

#include <fcntl.h>
#include <unistd.h>

#include "panvk_device.h"

static inline int
panvk_kbase_raw_fd(struct panvk_device *dev)
{
   (void)dev;

   static int fd = -1;

   if (fd < 0)
      fd = open("/dev/mali0", O_RDWR | O_CLOEXEC);

   return fd;
}

#endif /* PANVK_KBASE_FD_H */
