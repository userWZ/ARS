import pandapower.networks as pn

net = pn.case24_ieee_rts()
print(net)
print(net.gen)