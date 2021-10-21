import pandapower.networks as pn
import pandapower as pp
import pandas as pd
import numpy as np

net = pn.case_ieee30()
pp.runpp(net)
print(net)

#
# net.gen.to_csv('./IEEE30/gen.csv', sep=',', header=True, index=True)
# net.trafo.to_csv('./IEEE30/transformer.csv', sep=',', header=True, index=True)
# net.line.to_csv('./IEEE30/line.csv', sep=',', header=True, index=True)
# net.bus.to_csv('./IEEE30/bus.csv', sep=',', header=True, index=True)
# net.load.to_csv('./IEEE30/load.csv', sep=',', header=True, index=True)
# net.ext_grid.to_csv('./IEEE30/ext_grid.csv', sep=',', header=True, index=True)
