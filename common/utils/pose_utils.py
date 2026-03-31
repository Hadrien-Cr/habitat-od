from dataclasses import dataclass
import math


def rotate_vector(dx: float, dz: float, yaw: float) -> tuple[float, float]:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    rotated_dx = dx * cos_yaw - dz * sin_yaw
    rotated_dz = dx * sin_yaw + dz * cos_yaw

    return rotated_dx, rotated_dz



@dataclass
class DiscretizedAgentPose:
    idx_x: int
    idx_z: int
    idx_yaw: int
    idx_pitch: int
    yaw_bins: int
    pitch_bins: int

    def __hash__(self) -> int:
        return hash((self.idx_x, self.idx_z, self.idx_yaw, self.idx_pitch))

    def __repr__(self) -> str:
        return f"DiscAgentPose(x={self.idx_x}, z={self.idx_z}, yaw={self.idx_yaw}, h={self.idx_pitch})"

    def get_neigbhoring_states(
        self,
        grid_reachable_positions: list[tuple[int, int]],
        deltas: dict[str, tuple[int,int,int,int] | tuple[float, float, float, float]]
    ) -> list[tuple[str, "DiscretizedAgentPose"]]:
        """Get neighboring states of the given agent pose. Uses gridSnapping for handling diagonal movements"""
        neighbors = []

        for action_name, (dx, dz, dyaw, dpitch) in deltas.items():
            new_idx_yaw = int(self.idx_yaw + math.copysign(dyaw, 1)) % self.yaw_bins 
            new_idx_pitch = int(self.idx_yaw + math.copysign(dpitch, 1))
            
            rotated_dx, rotated_dz = rotate_vector(dx, dz, 360 * self.idx_yaw/self.yaw_bins)
            
            new_idx_x = int(self.idx_x + math.copysign(rotated_dx, 1))
            new_idx_z = int(self.idx_z + math.copysign(rotated_dz, 1))

            if (new_idx_x, new_idx_z) in grid_reachable_positions and 0 <= new_idx_pitch < self.pitch_bins:
                neigbor = DiscretizedAgentPose(
                    idx_x=new_idx_x,
                    idx_z=new_idx_z,
                    idx_yaw=new_idx_yaw,
                    idx_pitch=new_idx_pitch,
                    yaw_bins=self.yaw_bins,
                    pitch_bins=self.pitch_bins
                )
                neighbors.append((action_name, neigbor))

        return neighbors

    def __lt__(self, other: "DiscretizedAgentPose") -> bool:
        return (self.idx_x, self.idx_z, self.idx_yaw, self.idx_pitch) < (
            other.idx_x,
            other.idx_z,
            other.idx_yaw,
            other.idx_pitch,
        )

    def __eq__(self, other: "DiscretizedAgentPose") -> bool:
        return (self.idx_x, self.idx_z, self.idx_yaw, self.idx_pitch) == (
            other.idx_x,
            other.idx_z,
            other.idx_yaw,
            other.idx_pitch,
        )

def teleport_agent_pose(
    controller,
    target_pose: DiscretizedAgentPose,
) -> None:
    curr_pose = get_discrete_pose(controller)

    if curr_pose != target_pose:    
        y = controller.last_event.metadata["agent"]["position"]["y"]
        (x,z,yaw,pitch) = from_discrete_pose(
            target_pose,
            controller.grid_size,
            controller.grid_bounds,
            min_pitch=controller.min_pitch,
            max_pitch=controller.max_pitch        
        ) 

        try:
            controller.step(
                action="TeleportFull",
                position=dict(x=x, y=y, z=z),
                rotation=dict(x=0, y= yaw,z=0),
                standing=True,
                horizon=pitch,
                raise_for_failure=True,
            )
            controller.step(
                action="Pass",
                raise_for_failure=True,
            )
        except Exception as e:
            print(
                f"Teleportation failed at x:{x}, y:{y}, z:{z} with error {str(e)[0:70]}, retrying with forceAction=True"
            )
            controller.step(
                action="TeleportFull",
                position=dict(x=x, y=y, z=z),
                rotation=dict(x=0, y=yaw, z=0),
                standing=True,
                horizon=pitch,
                raise_for_failure=True,
                forceAction=True,
            )
            controller.step(
                action="Pass",
                raise_for_failure=True,
            )

def from_discrete_pose(
    pose: DiscretizedAgentPose,
    grid_size: float, 
    grid_bounds: tuple[float, float, float, float],
    min_pitch: float,
    max_pitch: float
) -> tuple[float, float, float, float]:
    grid_min_x, grid_max_x, grid_min_z, grid_max_z = grid_bounds
    
    x = grid_min_x + pose.idx_x * grid_size
    z = grid_min_z + pose.idx_z * grid_size
    pitch = (pose.idx_pitch/pose.pitch_bins * (max_pitch - min_pitch) + min_pitch)
    yaw  = 360 * pose.idx_yaw / pose.yaw_bins

    return x,z, yaw, pitch


def get_discrete_pose( controller, ) -> DiscretizedAgentPose:
    grid_min_x, grid_max_x, grid_min_z, grid_max_z = controller.grid_bounds
    
    md = controller.last_event.metadata
    agent_x, _, agent_z = tuple(md["agent"]["position"].values())
    _, yaw, _ = tuple(md["agent"]["rotation"].values())
    pitch = md["agent"]["cameraHorizon"]

    return DiscretizedAgentPose(
        idx_x=round((agent_x - grid_min_x) / controller.grid_size),
        idx_z=round((agent_z - grid_min_z) / controller.grid_size),
        idx_yaw=round(controller.yaw_bins * (yaw % 360) / 360),
        idx_pitch=round(controller.pitch_bins * (pitch + controller.min_pitch) / (controller.max_pitch - controller.min_pitch)),
        yaw_bins=controller.yaw_bins,
        pitch_bins=controller.pitch_bins
    )



