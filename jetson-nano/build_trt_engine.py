import os
import glob
import argparse
import numpy as np
import tensorrt as trt

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    _PYCUDA_OK = True
except ImportError:
    _PYCUDA_OK = False


class FingerprintCalibrator(trt.IInt8EntropyCalibrator2):

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __init__(self, calib_dir, cache_file, batch_size=8):
        super().__init__()
        if not _PYCUDA_OK:
            raise RuntimeError("pycuda is required for INT8: pip install pycuda")
        exts = ("*.bmp", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
        self.paths = [p for e in exts for p in glob.glob(os.path.join(calib_dir, "**", e), recursive=True)]
        if not self.paths:
            raise FileNotFoundError("No images found in: " + calib_dir)
        print("[Calibrator] %d images found." % len(self.paths))
        self.cache_file  = cache_file
        self.batch_size  = batch_size
        self.current_idx = 0
        self.device_buf  = cuda.mem_alloc(batch_size * 3 * 224 * 224 * 4)

    def _load_batch(self):
        from PIL import Image
        paths = self.paths[self.current_idx:self.current_idx + self.batch_size]
        if not paths:
            return None
        batch = []
        for p in paths:
            img = np.array(Image.open(p).convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
            img = (img.transpose(2, 0, 1) - self._MEAN) / self._STD
            batch.append(img)
        while len(batch) < self.batch_size:
            batch.append(batch[-1])
        self.current_idx += self.batch_size
        return np.ascontiguousarray(np.stack(batch), dtype=np.float32)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        batch = self._load_batch()
        if batch is None:
            return None
        cuda.memcpy_htod(self.device_buf, batch)
        return [int(self.device_buf)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print("[Calibrator] Loading cache: " + self.cache_file)
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print("[Calibrator] Writing cache: " + self.cache_file)
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_engine(onnx_path, output_path, min_batch, opt_batch, max_batch,
                 precision="fp16", calib_dir=None, calib_cache="vu_int8.cache",
                 calib_batch=8):
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()

    workspace_bytes = 4 * (1 << 30)
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        if not calib_dir:
            raise ValueError("--calib-dir is required for INT8.")
        # Calibration batch size is independent of the engine's optimization
        # profile, so the same cache can be reused across b1 / b16 / dynamic builds.
        config.int8_calibrator = FingerprintCalibrator(calib_dir, calib_cache, batch_size=calib_batch)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name,
        min=(min_batch, 3, 224, 224),
        opt=(opt_batch, 3, 224, 224),
        max=(max_batch, 3, 224, 224),
    )
    config.add_optimization_profile(profile)

    if min_batch == opt_batch == max_batch:
        print("Building %s engine (static batch=%d) ..." % (precision.upper(), opt_batch))
    else:
        print("Building %s engine (dynamic batch %d-%d, opt=%d) ..." % (precision.upper(), min_batch, max_batch, opt_batch))

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)

    size_mb = os.path.getsize(output_path) / 1024 ** 2
    print("Engine saved -> %s (%.1f MB)" % (output_path, size_mb))


def _suffixed(path, suffix):
    """Insert `_suffix` before the file extension, e.g. vu.engine -> vu_b1.engine"""
    root, ext = os.path.splitext(path)
    return "%s_%s%s" % (root, suffix, ext)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("onnx",   help="Path to the ONNX file")
    cli.add_argument("output", help="Output .engine path. With --mode all, used as a base "
                                     "name; suffixes _dynamic/_b1/_b16 are inserted before "
                                     "the extension.")
    cli.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    cli.add_argument("--mode", choices=["dynamic", "b1", "b16", "all"], default="dynamic",
                     help="dynamic: build with --min/opt/max-batch (default 1/8/16). "
                          "b1: static batch-1 engine. b16: static batch-16 engine. "
                          "all: build dynamic + b1 + b16 in one go.")
    cli.add_argument("--min-batch", type=int, default=1)
    cli.add_argument("--opt-batch", type=int, default=8)
    cli.add_argument("--max-batch", type=int, default=16)
    cli.add_argument("--calib-dir",   default=None, help="Calibration image dir (INT8 only)")
    cli.add_argument("--calib-cache", default="vu_int8.cache")
    cli.add_argument("--calib-batch", type=int, default=8,
                     help="Batch size used during INT8 calibration. Independent of the "
                          "engine's batch profile, so it stays fixed across b1/b16/dynamic builds.")
    args = cli.parse_args()

    if args.mode == "all":
        jobs = [
            ("dynamic", args.min_batch, args.opt_batch, args.max_batch, _suffixed(args.output, "dynamic")),
            ("b1",  1,  1,  1,  _suffixed(args.output, "b1")),
            ("b16", 16, 16, 16, _suffixed(args.output, "b16")),
        ]
    elif args.mode == "b1":
        jobs = [("b1", 1, 1, 1, args.output)]
    elif args.mode == "b16":
        jobs = [("b16", 16, 16, 16, args.output)]
    else:  # dynamic
        jobs = [("dynamic", args.min_batch, args.opt_batch, args.max_batch, args.output)]

    for name, mn, op, mx, out_path in jobs:
        print("\n--- [%s] ---" % name)
        build_engine(
            onnx_path   = args.onnx,
            output_path = out_path,
            min_batch   = mn,
            opt_batch   = op,
            max_batch   = mx,
            precision   = args.precision,
            calib_dir   = args.calib_dir,
            calib_cache = args.calib_cache,
            calib_batch = args.calib_batch,
        )
