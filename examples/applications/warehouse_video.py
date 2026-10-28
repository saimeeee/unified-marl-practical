"""CPU warehouse demo: packaged MFVFD -> InventoryEnv -> logged MP4 replay.

Run this file with the project's Python interpreter. No experiment starts on import.
The renderer displays exact recorded transitions; it never invents routing decisions.
"""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[2]
VIDEO_DEPS = PROJECT / ".work" / "scheduling-video-deps"
if VIDEO_DEPS.is_dir():
    sys.path.insert(0, str(VIDEO_DEPS))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sample_scene(directory, updates, seed):
    import numpy as np
    import yaml
    directory.mkdir()
    rows = []
    for index, (warehouse, sku, mean, initial, lead) in enumerate([
        ("WH1", "SKU1", 10, 26, 1), ("WH1", "SKU2", 7, 20, 2),
        ("WH2", "SKU1", 13, 31, 1), ("WH2", "SKU2", 9, 24, 2),
    ]):
        rows.append(dict(agent_id=f"{warehouse}_{sku}", warehouse_id=warehouse, sku_id=sku,
                         initial_stock=initial, capacity=100, mean_demand=mean,
                         lead_time=lead, unit_price=10, unit_cost=4,
                         holding_cost=0.15, shortage_cost=5))
    with (directory / "warehouse_agents.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    generator = np.random.default_rng(seed)
    with (directory / "warehouse_demand.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["period", "agent_id", "demand"])
        writer.writeheader()
        for period in range(24):
            for index, row in enumerate(rows):
                mean = row["mean_demand"] * (1 + .2 * math.sin((period + index) * math.pi / 6))
                writer.writerow(dict(period=period, agent_id=row["agent_id"],
                                     demand=int(generator.poisson(mean))))
    config = dict(schema_version=1, application="warehouse", backend="inventory",
                  algorithm="mfvfd", seed=seed, device="cpu", execution="local",
                  env_args=dict(agents_file="warehouse_agents.csv", demand_file="warehouse_demand.csv",
                                horizon=24, backlog=False, reward_scale=1000.),
                  training=dict(t_max=24 * (updates + 10), max_updates=updates,
                                batch_size=32, buffer_size=5000, num_threads=1,
                                num_groups=2, n_heads=2, embed_dim=64, hidden_dim=64,
                                hmf_steps_per_update=1, hmf_num_batch=1, hmf_norm_rewards=False),
                  output={})
    path = directory / "warehouse.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def rollout(checkpoint, config):
    import numpy as np
    from unified_marl.applications import warehouse
    from unified_marl.envs.inventory import InventoryEnv
    policy = warehouse.load_policy(checkpoint, config=config, device="cpu")
    policy.reset()  # Required at every new episode, including recurrent policies.
    environment = InventoryEnv(**dict(config["env_args"], seed=config["seed"]))
    records = []
    try:
        environment.reset()
        for period in range(environment.episode_limit):
            state = dict(period=period, agents=[dict(
                agent_id=row["agent_id"], stock=float(environment.stock[i]),
                backlog=float(environment.backlog[i]), last_demand=float(environment.last_demand[i]),
                pipeline=[dict(arrives_in=due-period, quantity=float(quantity))
                          for due, quantity in environment.pipeline[i]])
                for i, row in enumerate(environment.rows)])
            decision = policy.act(state, environment.get_avail_actions())
            before = environment.stock.copy()
            outstanding = [sum(quantity for _, quantity in pipeline) for pipeline in environment.pipeline]
            reward, done, info = environment.step(np.asarray(decision["raw_actions"], dtype=np.int64))
            for i, row in enumerate(info["decisions"]):
                if not np.isclose(row["executed_order"], decision["decisions"][i]["order_quantity"], atol=1e-4):
                    raise AssertionError("Application decision and simulator order differ")
                if not np.isclose(before[i] + row["arrivals"] - row["sales"], row["ending_stock"]):
                    raise AssertionError("Inventory conservation failed")
                if not -1e-6 <= row["executed_order"] <= environment.capacity[i]-before[i]-outstanding[i]+1e-6:
                    raise AssertionError("Capacity reservation failed")
                row["starting_stock"] = float(before[i])
                row["in_transit"] = float(sum(q for _, q in environment.pipeline[i]))
                row["capacity"] = float(environment.capacity[i])
            records.append(dict(period=period, reward=float(reward), terminated=bool(done),
                                decisions=info["decisions"], totals=environment.get_stats()))
            if done:
                if period != environment.episode_limit - 1:
                    raise AssertionError("Unexpected episode termination")
                break
        if len(records) != environment.episode_limit or not records[-1]["terminated"]:
            raise AssertionError("Incomplete recorded episode")
        return records
    finally:
        environment.close()


def render(records, output, fps, seconds_per_period, label, updates):
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg
    width, height = 1280, 720
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if not font_path.is_file():
        raise FileNotFoundError("This Windows demo requires Microsoft YaHei: C:/Windows/Fonts/msyh.ttc")
    fonts = {n: ImageFont.truetype(str(font_path), n) for n in [17, 19, 21, 24, 26, 34]}
    blue, ink, muted = "#00559D", "#172434", "#586B80"
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
               "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    frames = 0
    with output.with_suffix(".encoder.log").open("wb") as errors:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for step, record in enumerate(records):
                for frame in range(max(1, round(fps * seconds_per_period))):
                    image = Image.new("RGB", (width, height), "#F8FAFD")
                    draw = ImageDraw.Draw(image)
                    def text(x, y, value, size=21, color=ink):
                        draw.text((x, y), str(value), font=fonts[size], fill=color)
                    text(38, 22, "多智能体仓储库存补货", 34, blue)
                    text(39, 73, f"{label}    CPU    MFVFD    第 {step+1:02d}/{len(records)} 期", 24)
                    text(39, 113, f"实际模型决策与环境记录    已训练 {updates} 次更新    绿色圆点表示当期到货", 19, muted)
                    draw.line((38, 150, 1240, 150), fill="#C8D8E8", width=2)
                    for i, decision in enumerate(record["decisions"]):
                        y = 172 + i * 109
                        text(39, y, decision["agent_id"], 24, blue)
                        text(39, y+37, f"下单 {decision['executed_order']:.1f}", 21)
                        text(213, y, f"当期到货 {decision['arrivals']:.1f}", 21)
                        draw.line((222, y+48, 414, y+48), fill="#8AA9C3", width=4)
                        draw.polygon([(415, y+48), (402, y+40), (402, y+56)], fill="#8AA9C3")
                        if decision["arrivals"] > 0:
                            x = 227 + round(174 * frame / max(1, round(fps*seconds_per_period)-1))
                            draw.ellipse((x-6, y+42, x+6, y+54), fill="#218A66")
                        text(441, y, f"期末库存 {decision['ending_stock']:.1f} / {decision['capacity']:.0f}", 21)
                        draw.rectangle((443, y+40, 682, y+65), fill="#DFE8F1")
                        fraction = max(0., min(1., decision["ending_stock"]/decision["capacity"]))
                        if fraction:
                            draw.rectangle((443, y+40, 443+239*fraction, y+65), fill=blue)
                        text(729, y, f"需求 {decision['demand']:.0f}   满足 {decision['sales']:.1f}", 21)
                        shortage = decision["demand"] - decision["sales"]
                        text(729, y+37, f"未满足 {shortage:.1f}", 21, "#B84632" if shortage>0 else "#218A66")
                        text(1052, y, f"在途 {decision['in_transit']:.1f}", 21)
                        text(1052, y+37, f"利润 {decision['profit']:.1f}", 21)
                        draw.line((39, y+91, 1238, y+91), fill="#DDE5ED", width=1)
                    totals = record["totals"]
                    demand = sum(d["demand"] for r in records[:step+1] for d in r["decisions"])
                    fill = totals["sales"]/demand if demand else 0
                    text(39, 620, f"累计销售 {totals['sales']:.1f}    累计需求 {demand:.0f}    满足率 {fill:.1%}    累计利润 {totals['profit']:.1f}", 24, blue)
                    text(39, 667, "合成需求下的短训练演示；未验证收敛或真实业务收益。画面为库存与补货，不包含车辆路径。", 19, muted)
                    encoder.stdin.write(image.tobytes())
                    if frames == 0:
                        image.save(output.with_name("preview.png"))
                    frames += 1
            encoder.stdin.close()
            if encoder.wait(timeout=120):
                raise RuntimeError("MP4 encoder failed; inspect the encoder log")
        finally:
            if encoder.poll() is None:
                encoder.kill()
                encoder.wait()
    # Decode every frame, rather than only checking that a file was created.
    probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                            "-i", str(output), "-f", "null", "-"], capture_output=True, timeout=120)
    if probe.returncode:
        raise RuntimeError("MP4 decode failed: " + probe.stderr.decode(errors="replace"))
    return dict(frames=frames, fps=fps, duration_seconds=frames/fps, width=width, height=height,
                codec="h264", pixel_format="yuv420p", full_decode_passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=120, help="Bounded short demo budget, not convergence")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", type=Path, help="Optional real warehouse YAML; recording supports up to four agents")
    parser.add_argument("--checkpoint", type=Path, help="Load an existing application policy.pt and skip training")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seconds-per-period", type=float, default=1.)
    args = parser.parse_args()
    if args.updates < 1 or not 1 <= args.fps <= 60 or not .1 <= args.seconds_per_period <= 5:
        parser.error("updates>=1, fps=1..60, seconds-per-period=.1..5 required")
    import torch
    import unified_marl
    from unified_marl.applications import warehouse, load_config, validate
    try:
        import PIL
        import imageio_ffmpeg
    except ImportError as error:
        raise SystemExit("Install video dependencies into .work/scheduling-video-deps; see WAREHOUSE_VIDEO.md") from error
    # Check codec path and font before starting even a short training job.
    if not Path(imageio_ffmpeg.get_ffmpeg_exe()).is_file() or not Path("C:/Windows/Fonts/msyh.ttc").is_file():
        raise SystemExit("FFmpeg or Microsoft YaHei font missing")
    output = (args.output_dir or PROJECT / "results" / ("warehouse-video-"+datetime.now().strftime("%Y%m%d-%H%M%S-%f"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    if args.checkpoint:
        checkpoint = args.checkpoint.resolve()
        saved = json.loads((checkpoint.parent / "application.json").read_text(encoding="utf-8"))
        config = load_config(args.config) if args.config else saved["config"]
        update_count = int(json.loads((checkpoint.parent/"application-result.json").read_text(encoding="utf-8"))["updates"])
        training = None
    else:
        path = args.config.resolve() if args.config else sample_scene(output/"scene", args.updates, args.seed)
        config = load_config(path, overrides={"device":"cpu", "execution":"local", "training.max_updates":args.updates,
                                            "training.t_max":max(10000, args.updates * 100)})
        checkpoint = None
        training = None
        update_count = args.updates
    report = validate(config)
    if report["algorithm"] != "wang_hmf/mfvfd":
        raise ValueError("This MFVFD-labelled demo requires algorithm: mfvfd; use another renderer label for other algorithms")
    if report["environment"] != "inventory" or config["env_args"].get("backlog", False):
        raise ValueError("This renderer requires inventory with backlog=false for explicit demand accounting")
    if not 1 <= len(report["metadata"]["agents"]) <= 4:
        raise ValueError("Renderer supports one to four agents; adapt layout for larger scenarios")
    if checkpoint is None:
        print(f"Training {config['algorithm']} on CPU, max_updates={args.updates}", flush=True)
        result = warehouse.train(config, output_dir=output/"model")
        training = asdict(result)
        checkpoint = Path(result.checkpoint)
        update_count = result.updates
        if update_count != args.updates or result.raw["parameter_delta_l2"] <= 0:
            raise AssertionError("Expected the requested number of effective parameter updates")
        if not all(math.isfinite(float(x)) for x in result.metrics.values()):
            raise AssertionError("Non-finite training loss")
    print("Recording exact policy decisions and simulator transitions", flush=True)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        records = rollout(checkpoint, config)
    finally:
        torch.set_num_threads(original_threads)
    write_json(output/"rollout.json", records)
    flat = [dict(period=record["period"], **decision) for record in records for decision in record["decisions"]]
    with (output/"rollout.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print("Rendering MP4 from recorded data", flush=True)
    label = "合成示例" if not args.config and not args.checkpoint else "用户配置场景"
    video = output/"warehouse_mfvfd.mp4"
    media = render(records, video, args.fps, args.seconds_per_period, label, update_count)
    manifest = dict(status="completed", python=sys.version, interpreter=sys.executable,
                    platform=sys.platform, torch=torch.__version__, unified_marl=unified_marl.__version__,
                    package_path=unified_marl.__file__, algorithm=report["algorithm"],
                    configuration=config, training=training, checkpoint=str(checkpoint),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    video=str(video), video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                    media=media, periods=len(records), totals=records[-1]["totals"],
                    elapsed_seconds=time.perf_counter()-start,
                    assertions=["finite losses / positive parameter change for new training",
                                "application/environment orders match", "inventory conservation",
                                "capacity reservation", "complete episode", "full MP4 decode"],
                    boundary="Synthetic short demo is not convergence, a held-out evaluation, or real business evidence; no routing model.")
    write_json(output/"manifest.json", manifest)
    print(json.dumps({"video":str(video), "data":str(output/"rollout.csv"),
                      "manifest":str(output/"manifest.json"), "updates":update_count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
