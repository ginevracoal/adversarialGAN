import fbm
import numpy as np
import matplotlib.pyplot as plot

H = 0.7
offslo = (1, 0.5)

x = np.linspace(1, 10, 1000)
fbm = fbm.generate(H, x, offset_slope = offslo)

xSegment = ( x[0], x[-1] )
ySegment = ( offslo[0] + offslo[1] * x[0], offslo[0] + offslo[1] * x[-1] )

plot.figure()
f1, = plot.plot(xSegment, ySegment)
f2, = plot.plot(*fbm)

plot.legend([f1, f2], ['Funzione lineare di primo grado', 'moto Browniano frattale'])

plot.show()
