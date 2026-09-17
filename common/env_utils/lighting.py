from habitat_sim.gfx import LightInfo, LightPositionModel

# DEFAULT_LIGHTING_KEY's own 4-direction rig (habitat-sim LightSetup.cpp::getDefaultLights) at 2x
# intensity, plus a soft overhead fill -- approximates ai2thor's brighter, evenly-lit look within
# this Phong path's constraints (no tonemapping, so it clips highlights instead of compressing
# them -- picked via scripts/gen_lighting_jpgs.py's lighting sweep).
AI2THOR_BRIGHT_LIGHTS = [
    LightInfo(vector=[0.0, -0.5, -0.5, 0.0], color=[0.6, 0.6, 0.6], model=LightPositionModel.Global),
    LightInfo(vector=[0.0, -0.5, 0.5, 0.0], color=[0.6, 0.6, 0.6], model=LightPositionModel.Global),
    LightInfo(vector=[-0.5, -0.5, 0.0, 0.0], color=[0.6, 0.6, 0.6], model=LightPositionModel.Global),
    LightInfo(vector=[0.5, -0.5, 0.0, 0.0], color=[0.6, 0.6, 0.6], model=LightPositionModel.Global),
    LightInfo(vector=[0.0, -1.0, 0.0, 0.0], color=[0.4, 0.4, 0.4], model=LightPositionModel.Global),
]
