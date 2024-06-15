import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from paths import find_repository_root

IMAGE_NAME = "mirandatz/morefun:dev_env"


@dataclass(frozen=True)
class MountPoint:
    host_path: str
    container_path: str
    mode: str


@dataclass(frozen=True)
class MorefunMountPoints:
    code: MountPoint
    dataset: MountPoint
    settings: MountPoint
    output: MountPoint


def extract_mount_points_from_settings(
    dataset_dir: Path,
    settings_path: Path,
    output_path: Path,
) -> MorefunMountPoints:
    settings_yaml = yaml.safe_load(settings_path.read_text())
    return MorefunMountPoints(
        code=MountPoint(find_repository_root(), Path("/app/code"), "ro"),
        dataset=MountPoint(
            dataset_dir, Path(settings_yaml["dataset"]["partitions_dir"]), "ro"
        ),
        settings=MountPoint(settings_path, Path("/app/settings.yaml"), "ro"),
        output=MountPoint(output_path, Path(settings_yaml["output"]["directory"]), ""),
    )


def make_docker_base_args(mount_points: MorefunMountPoints) -> list[str]:
    uid = os.getuid()
    gid = os.getgid()

    return [
        "docker",
        "run",
        "--rm",
        "--runtime=nvidia",
        "--shm-size=8gb",
        f"--user={uid}:{gid}",
        f"-v={mount_points.code.host_path}:{mount_points.code.container_path}:{mount_points.code.mode}",
        f"-v={mount_points.dataset.host_path}:{mount_points.dataset.container_path}:{mount_points.dataset.mode}",
        f"-v={mount_points.settings.host_path}:{mount_points.settings.container_path}:{mount_points.settings.mode}",
        f"-v={mount_points.output.host_path}:{mount_points.output.container_path}:{mount_points.output.mode}",
        f"--workdir={mount_points.code.container_path}",
        IMAGE_NAME,
    ]


def main() -> None:
    settings_path = (
        find_repository_root()
        / "morefun"
        / "experiments"
        / "v2"
        / "settings_template.yaml"
    )
    mount_points = extract_mount_points_from_settings(
        Path("host_fake_dataset_path"),
        settings_path,
        Path("fake_host_output_path"),
    )
    print(make_docker_base_args(mount_points))


if __name__ == "__main__":
    main()
