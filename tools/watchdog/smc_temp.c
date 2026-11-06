// Print CPU temperatures (°C) from the SMC of an Intel Mac, without root.
// Build: clang -O2 -framework IOKit -o smc_temp smc_temp.c
// Output: one line per readable key, "KEY value"
#include <IOKit/IOKitLib.h>
#include <stdio.h>
#include <string.h>

#define KERNEL_INDEX_SMC 2
#define SMC_CMD_READ_BYTES 5
#define SMC_CMD_READ_KEYINFO 9

typedef struct { char major, minor, build, reserved[1]; UInt16 release; } vers_t;
typedef struct { UInt16 version, length; UInt32 cpuPLimit, gpuPLimit, memPLimit; } plimit_t;
typedef struct { UInt32 dataSize, dataType; char dataAttributes; } keyinfo_t;
typedef struct {
    UInt32 key; vers_t vers; plimit_t pLimitData; keyinfo_t keyInfo;
    char result, status, data8; UInt32 data32; unsigned char bytes[32];
} smcdata_t;

static UInt32 str4(const char *s) { return (UInt32)s[0] << 24 | (UInt32)s[1] << 16 | (UInt32)s[2] << 8 | (UInt32)s[3]; }

static int call(io_connect_t c, smcdata_t *in, smcdata_t *out) {
    size_t sz = sizeof(smcdata_t);
    return IOConnectCallStructMethod(c, KERNEL_INDEX_SMC, in, sizeof(smcdata_t), out, &sz);
}

static int read_key(io_connect_t c, const char *key, double *val) {
    smcdata_t in, out;
    memset(&in, 0, sizeof in); memset(&out, 0, sizeof out);
    in.key = str4(key); in.data8 = SMC_CMD_READ_KEYINFO;
    if (call(c, &in, &out) != kIOReturnSuccess || out.result) return 0;
    UInt32 size = out.keyInfo.dataSize, type = out.keyInfo.dataType;
    in.keyInfo.dataSize = size; in.data8 = SMC_CMD_READ_BYTES;
    memset(&out, 0, sizeof out);
    if (call(c, &in, &out) != kIOReturnSuccess || out.result) return 0;
    if (type == str4("sp78") && size == 2) { *val = (signed char)out.bytes[0] + out.bytes[1] / 256.0; return 1; }
    if (type == str4("flt ") && size == 4) { float f; memcpy(&f, out.bytes, 4); *val = f; return 1; }
    return 0;
}

int main(void) {
    io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSMC"));
    if (!svc) { fprintf(stderr, "AppleSMC not found\n"); return 1; }
    io_connect_t c;
    if (IOServiceOpen(svc, mach_task_self(), 0, &c) != kIOReturnSuccess) { fprintf(stderr, "cannot open SMC\n"); return 1; }
    // CPU proximity, die, PECI and core keys used on Intel MacBook Pros
    const char *keys[] = {"TC0P", "TC0D", "TC0E", "TC0F", "TC0H", "TCXC", "TCSA", "TC1C", "TC2C", "TC3C", "TC4C",
                          "TC5C", "TC6C", "TC7C", "TC8C", "TB0T", "TH0x", "TG0P"};
    int n = 0;
    for (size_t i = 0; i < sizeof keys / sizeof *keys; i++) {
        double v;
        if (read_key(c, keys[i], &v) && v > 0 && v < 130) { printf("%s %.1f\n", keys[i], v); n++; }
    }
    IOServiceClose(c);
    return n ? 0 : 2;
}
