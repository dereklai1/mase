# Get entire constraint file at:

# Clocks
create_clock -name clk1 -period 2.0 [get_ports clk]

#
# Power Constraint to warn User if Design will possibly be over cards power limit, this assume the 2x4 PCIe AUX power is connected to the board.
#
set_operating_conditions -design_power_budget 160
