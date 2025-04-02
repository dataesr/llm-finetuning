from subprocess import Popen, PIPE, STDOUT, CalledProcessError
from app.logging import get_logger

logger = get_logger(__name__)


def run(command=list[str]):
    shell_cmd = " ".join(command)
    logger.info(f"RUN {shell_cmd}")

    process = Popen(shell_cmd, shell=True, stdout=PIPE, stderr=STDOUT)
    with process.stdout:
        try:
            for line in iter(process.stdout.readline, b""):
                print(line.decode("utf-8").strip())

        except (OSError, CalledProcessError) as exception:
            print(f"CMD FAILED: {str(exception)}")
            return False
    return True
