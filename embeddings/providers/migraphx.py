from typing import List


def get_migraphx_providers() -> List[str]:
    # MIGraphX-only; no ROCmExecutionProvider fallback with the migraphx wheel.
    return ["MIGraphXExecutionProvider"]
