/*
   Configure the latency timer to a small number

   Based on libftdi/examples/simple.c

   This program is distributed under the GPL, version 2
*/

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <ftdi.h>

int main(int argc, char **argv)
{
    struct ftdi_context *ftdi;
    int f = 0, i;
    int vid = 0x0;
    int pid = 0x0;
    int interface = INTERFACE_ANY;
    int retval = EXIT_FAILURE;
    unsigned char latency = 0;

    while ((i = getopt(argc, argv, "i:v:p:l:")) != -1) {
        switch (i) {
            case 'i': // 0=ANY, 1=A, 2=B, 3=C, 4=D
                interface = strtoul(optarg, NULL, 0);
                break;
            case 'v':
                vid = strtoul(optarg, NULL, 0);
                break;
            case 'p':
                pid = strtoul(optarg, NULL, 0);
                break;
            case 'l':
                latency = (unsigned char)strtoul(optarg, NULL, 0);
                break;
            default:
                fprintf(stderr, "usage: %s [-i interface] [-v vid] [-p pid] [-l latency]\n", *argv);
                exit(EXIT_FAILURE);
        }
    }

    if (latency == 0) {
        fprintf(stderr, "Please provide a valid latency value using -l option\n");
        exit(EXIT_FAILURE);
    }

    // Init
    if ((ftdi = ftdi_new()) == 0) {
        fprintf(stderr, "ftdi_new failed\n");
        return EXIT_FAILURE;
    }

    if (!vid && !pid && (interface == INTERFACE_ANY)) {
        ftdi_set_interface(ftdi, INTERFACE_ANY);
        struct ftdi_device_list *devlist;
        int res;
        if ((res = ftdi_usb_find_all(ftdi, &devlist, 0, 0)) < 0) {
            fprintf(stderr, "No FTDI with default VID/PID found\n");
            goto do_deinit;
        }
        if (res == 1) {
            f = ftdi_usb_open_dev(ftdi, devlist[0].dev);
        }
        ftdi_list_free(&devlist);
        if (res > 1) {
            fprintf(stderr, "%d Devices found, please select Device with VID/PID\n", res);
            goto do_deinit;
        }
        if (res == 0) {
            fprintf(stderr, "No Devices found with default VID/PID\n");
            goto do_deinit;
        }
    } else {
        // Select interface
        ftdi_set_interface(ftdi, interface);

        // Open device
        f = ftdi_usb_open(ftdi, vid, pid);
    }
    if (f < 0) {
        fprintf(stderr, "unable to open ftdi device: %d (%s)\n", f, ftdi_get_error_string(ftdi));
        exit(EXIT_FAILURE);
    }

    if (ftdi_get_latency_timer(ftdi, &latency) == 0) {
        printf("Got latency %i\n", latency);
        latency = (unsigned char)strtoul(optarg, NULL, 0); // Update latency from input argument
        f = ftdi_set_latency_timer(ftdi, latency);
        if (f == 0) {
            printf("Set latency to %i\n", latency);
            retval =  EXIT_SUCCESS;
        } else {
            fprintf(stderr, "Failed to set latency: %d (%s)\n", f, ftdi_get_error_string(ftdi));
        }
    }

    ftdi_usb_close(ftdi);

do_deinit:
    ftdi_free(ftdi);
    return retval;
}
