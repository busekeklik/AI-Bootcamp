import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Create the fuzzy control system
def create_ac_controller():
    # Define input/output variables
    temperature = ctrl.Antecedent(np.arange(15, 35, 1), 'temperature')
    humidity = ctrl.Antecedent(np.arange(30, 90, 1), 'humidity')
    ac_power = ctrl.Consequent(np.arange(0, 101, 1), 'ac_power')

    # Define membership functions
    temperature['cool'] = fuzz.trimf(temperature.universe, [15, 20, 25])
    temperature['moderate'] = fuzz.trimf(temperature.universe, [20, 25, 30])
    temperature['hot'] = fuzz.trimf(temperature.universe, [25, 30, 35])

    humidity['low'] = fuzz.trimf(humidity.universe, [30, 40, 50])
    humidity['medium'] = fuzz.trimf(humidity.universe, [40, 60, 70])
    humidity['high'] = fuzz.trimf(humidity.universe, [60, 75, 90])

    ac_power['low'] = fuzz.trimf(ac_power.universe, [0, 25, 50])
    ac_power['medium'] = fuzz.trimf(ac_power.universe, [25, 50, 75])
    ac_power['high'] = fuzz.trimf(ac_power.universe, [50, 75, 100])

    # Define fuzzy rules
    rule1 = ctrl.Rule(temperature['cool'] & humidity['low'], ac_power['low'])
    rule2 = ctrl.Rule(temperature['cool'] & humidity['medium'], ac_power['low'])
    rule3 = ctrl.Rule(temperature['cool'] & humidity['high'], ac_power['medium'])
    rule4 = ctrl.Rule(temperature['moderate'] & humidity['low'], ac_power['low'])
    rule5 = ctrl.Rule(temperature['moderate'] & humidity['medium'], ac_power['medium'])
    rule6 = ctrl.Rule(temperature['moderate'] & humidity['high'], ac_power['high'])
    rule7 = ctrl.Rule(temperature['hot'] & humidity['low'], ac_power['medium'])
    rule8 = ctrl.Rule(temperature['hot'] & humidity['medium'], ac_power['high'])
    rule9 = ctrl.Rule(temperature['hot'] & humidity['high'], ac_power['high'])

    # Create control system
    ac_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    return ctrl.ControlSystemSimulation(ac_ctrl)

# Function to get AC power recommendation
def get_ac_power(temp, hum):
    ac = create_ac_controller()
    ac.input['temperature'] = temp
    ac.input['humidity'] = hum
    ac.compute()
    return ac.output['ac_power']

# Example usage
temp = 28  # 28°C
humidity = 65  # 65%
power = get_ac_power(temp, humidity)
print(f"Recommended AC power: {power:.1f}%")