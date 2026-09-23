#!/usr/bin/env python3
"""Generate printable ArUco markers at an exact physical size.

  python3 make_markers.py                        # ids 0-3, 60 mm, A4
  python3 make_markers.py --size-mm 40 --ids 0 1 2 3 4 5
  python3 make_markers.py --dict DICT_5X5_100 --ids 7

Why the size matters more than anything else here: a single square tag gives
a full 6-DOF pose only because its physical side length is known.  Pose
distance scales linearly with that number, so a marker printed 5% small puts
the object 5% closer than it is - at 400 mm that is a 20 mm error, which is
the difference between a grasp and a miss.

Printers lie about scale.  "Fit to page", "shrink oversized pages" and
borderless modes all resize silently, so every sheet carries a 100 mm ruler:
measure it after printing, and if it is not 100 mm, either reprint at 100%
scale or measure the marker itself and pass the real number to the detector
as marker_size_m.  Measuring the printed marker is always more trustworthy
than trusting the printer.
"""
import argparse
import sys

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError:
    sys.exit('matplotlib is needed to lay the page out at an exact scale')

MM = 1.0 / 25.4          # millimetres to inches


def build(dict_name, ids, size_mm, out, per_page, quiet_cells):
    if not hasattr(cv2.aruco, dict_name):
        sys.exit('unknown dictionary {}. Try DICT_4X4_50, DICT_5X5_100, '
                 'DICT_6X6_250 or DICT_APRILTAG_36h11'.format(dict_name))
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))

    # Render each marker far larger than it will print, so the printer is the
    # only thing limiting edge sharpness.  Blurred edges cost corner accuracy,
    # and corner accuracy is what the pose is made of.
    px = 1200

    with PdfPages(out) as pdf:
        for chunk in [ids[i:i + per_page] for i in range(0, len(ids),
                                                         per_page)]:
            fig = plt.figure(figsize=(210 * MM, 297 * MM))   # A4 portrait
            fig.subplots_adjust(0, 0, 1, 1)

            for k, mid in enumerate(chunk):
                img = cv2.aruco.generateImageMarker(d, int(mid), px)

                # A quiet zone is not decoration: the detector looks for a
                # black square on a light background, and without a white
                # border a marker printed to the edge of dark bench or dark
                # tape simply is not found.
                cell = px // (d.markerSize + 2)
                pad = quiet_cells * cell
                sheet = np.full((px + 2 * pad, px + 2 * pad), 255, np.uint8)
                sheet[pad:pad + px, pad:pad + px] = img

                total_mm = size_mm * (px + 2 * pad) / float(px)
                col = k % 2
                row = k // 2
                x0 = 15 + col * 100
                y0 = 250 - row * 95

                ax = fig.add_axes([x0 * MM / (210 * MM),
                                   (y0 - total_mm) * MM / (297 * MM),
                                   total_mm * MM / (210 * MM),
                                   total_mm * MM / (297 * MM)])
                ax.imshow(sheet, cmap='gray', vmin=0, vmax=255,
                          interpolation='nearest')
                ax.axis('off')

                cap = fig.add_axes([x0 * MM / (210 * MM),
                                    (y0 - total_mm - 8) * MM / (297 * MM),
                                    90 * MM / (210 * MM),
                                    7 * MM / (297 * MM)])
                cap.axis('off')
                cap.text(0, 0.5, '{}  id={}   {:.0f} mm'.format(
                    dict_name, mid, size_mm), fontsize=8, va='center')

            # The scale check.  Everything above is only as true as this line.
            ruler = fig.add_axes([15 * MM / (210 * MM), 25 * MM / (297 * MM),
                                  100 * MM / (210 * MM), 20 * MM / (297 * MM)])
            ruler.set_xlim(0, 100)
            ruler.set_ylim(0, 20)
            ruler.axis('off')
            ruler.plot([0, 100], [10, 10], 'k-', lw=1)
            for t in range(0, 101, 10):
                ruler.plot([t, t], [10, 15], 'k-', lw=1)
                ruler.text(t, 16, str(t), fontsize=6, ha='center')
            ruler.text(0, 4,
                       'MEASURE ME: this line is exactly 100 mm. If it is '
                       'not, the markers are not {:.0f} mm either - reprint '
                       'at 100% scale, or measure a marker\'s black square '
                       'and pass that as marker_size_m.'.format(size_mm),
                       fontsize=6.5, va='center', wrap=True)

            head = fig.add_axes([15 * MM / (210 * MM), 275 * MM / (297 * MM),
                                 180 * MM / (210 * MM), 15 * MM / (297 * MM)])
            head.axis('off')
            head.text(0, 0.5,
                      'OpenArm markers - {}, {:.0f} mm black square, '
                      '{} cell quiet zone.  Print at 100% scale, no fitting.'
                      .format(dict_name, size_mm, quiet_cells),
                      fontsize=9, va='center')

            pdf.savefig(fig)
            plt.close(fig)

    print('wrote {}'.format(out))
    print()
    print('  dictionary   : {}'.format(dict_name))
    print('  ids          : {}'.format(', '.join(str(i) for i in ids)))
    print('  marker size  : {:.0f} mm  (the BLACK SQUARE, not the white '
          'border)'.format(size_mm))
    print()
    print('  Print at 100% scale.  Then measure the 100 mm ruler, and give')
    print('  the detector the real size:')
    print('     ros2 param set /marker_detector marker_size_m {:.3f}'
          .format(size_mm / 1000.0))
    print()
    print('  Stick it flat.  A marker bent around a curved object has no')
    print('  single plane, and the pose it yields is meaningless.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dict', default='DICT_4X4_50')
    ap.add_argument('--ids', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--size-mm', type=float, default=60.0)
    ap.add_argument('--per-page', type=int, default=4)
    ap.add_argument('--quiet-cells', type=int, default=1)
    ap.add_argument('-o', '--out', default='aruco_markers.pdf')
    a = ap.parse_args()
    build(a.dict, a.ids, a.size_mm, a.out, a.per_page, a.quiet_cells)


if __name__ == '__main__':
    main()
