import os
import tempfile
from typing import Optional

import fire
from demuxai.settings.main import Settings


default_config_file = os.getenv("DEMUXAI_CONFIG_FILE", "config.yml")


def run(
    config_file: Optional[str] = default_config_file,
    listen: Optional[str] = None,
    port: Optional[int] = None,
):
    """Start the DemuxAI API server"""
    import uvicorn

    if config_file != default_config_file:
        os.environ.update(DEMUXAI_CONFIG_FILE=config_file)

    settings = Settings.load(config_file)
    listen = listen or settings.listen
    port = port or settings.port

    # create a temp file to act as a .env file containing an environment var with the path to
    # the config file
    with tempfile.NamedTemporaryFile(mode="w") as env:
        env.write(f"DEMUXAI_CONFIG_FILE={os.path.realpath(config_file)}")
        env.flush()

        uvicorn.run(
            "demuxai.api:api",
            host=listen,
            port=port,
            log_level="debug",
            reload=False,
            env_file=env.name,
        )


def main():
    fire.Fire(run)


if __name__ == "__main__":
    main()
