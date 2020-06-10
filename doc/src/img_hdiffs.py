import fbm
import numpy as np
import matplotlib.pyplot as plot

offslo = (0, 1)
x = np.linspace(1, 10, 100)

plot.figure(1)
H = 0.1
fbm_distr = fbm.generate(H, x, offset_slope = offslo)
f1, = plot.plot(*fbm_distr, color='b', linewidth=2)
plot.legend([f1], ['H = ' + str(H)])

plot.figure(2)
H = 0.5
fbm_distr = fbm.generate(H, x, offset_slope = offslo)
f1, = plot.plot(*fbm_distr, color='r', linewidth=2)
plot.legend([f1], ['H = ' + str(H)])

plot.figure(3)
H = 0.9
fbm_distr = fbm.generate(H, x, offset_slope = offslo)
f1, = plot.plot(*fbm_distr, color='g', linewidth=2)
plot.legend([f1], ['H = ' + str(H)])

plot.show()
