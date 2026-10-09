import argparse
import base64
import json
import sys
from dataclasses import dataclass

import cv2
import numpy as np


IMAGE_FIELDS = {"shadeImage", "cutoutImage"}


@dataclass
class SolveResult:
    point_x: int
    edge_x: int
    dark_x: int
    edge_score: float

    @property
    def confident(self) -> bool:
        return abs(self.edge_x - self.dark_x) <= 2


def decode_image(b64: str, flags: int) -> np.ndarray:
    if "," in b64[:64]:
        b64 = b64.split(",", 1)[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), flags)


def piece_mask(cutout: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(cutout[:, :, 3], 127, 255, cv2.THRESH_BINARY)
    return mask


def match_by_edges(shade_gray: np.ndarray, mask: np.ndarray, point_y: int) -> tuple[int, float]:
    h = mask.shape[0]
    piece_edges = cv2.Canny(mask, 100, 200)
    strip = cv2.GaussianBlur(shade_gray[point_y:point_y + h, :], (3, 3), 0)
    strip_edges = cv2.Canny(strip, 50, 150)
    result = cv2.matchTemplate(strip_edges, piece_edges, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, _) = cv2.minMaxLoc(result)
    return x, float(score)


def match_by_darkness(shade_gray: np.ndarray, mask: np.ndarray, point_y: int) -> int:
    h, w = mask.shape
    inside = mask > 0
    strip = shade_gray[point_y:point_y + h, :].astype(np.float32)
    means = [strip[:, x:x + w][inside].mean() for x in range(strip.shape[1] - w + 1)]
    return int(np.argmin(means))


def solve(shade_b64: str, cutout_b64: str, point_y: int) -> SolveResult:
    shade_gray = decode_image(shade_b64, cv2.IMREAD_GRAYSCALE)
    cutout = decode_image(cutout_b64, cv2.IMREAD_UNCHANGED)
    if cutout is None or cutout.ndim != 3 or cutout.shape[2] != 4:
        raise ValueError("cutoutImage must be a PNG with an alpha channel")
    mask = piece_mask(cutout)
    edge_x, edge_score = match_by_edges(shade_gray, mask, point_y)
    dark_x = match_by_darkness(shade_gray, mask, point_y)
    return SolveResult(point_x=edge_x, edge_x=edge_x, dark_x=dark_x, edge_score=edge_score)


def solve_response(payload: dict) -> SolveResult:
    data = payload.get("data", payload)
    return solve(data["shadeImage"], data["cutoutImage"], int(data["pointY"]))


def save_debug_image(payload: dict, result: SolveResult, path: str) -> None:
    data = payload.get("data", payload)
    shade = decode_image(data["shadeImage"], cv2.IMREAD_COLOR)
    mask = piece_mask(decode_image(data["cutoutImage"], cv2.IMREAD_UNCHANGED))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(shade, contours, -1, (0, 255, 0), 1, offset=(result.point_x, int(data["pointY"])))
    cv2.imwrite(path, cv2.resize(shade, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))


def run_offline(args: argparse.Namespace) -> int:
    with open(args.file, encoding="utf-8") as f:
        payload = json.load(f)
    result = solve_response(payload)
    print(json.dumps({**result.__dict__, "confident": result.confident}))
    if args.debug_image:
        save_debug_image(payload, result, args.debug_image)
    return 0


def run_online(args: argparse.Namespace) -> int:
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})
    if args.cookie:
        session.headers["Cookie"] = args.cookie

    output = {}
    for attempt in range(1, args.attempts + 1):
        payload = session.request(args.get_method, args.get_url, timeout=15).json()
        result = solve_response(payload)
        output = build_online_output(payload, result, session, args.offset)
        if result.confident:
            break
        print(f"[{attempt}] low confidence, fetching a new captcha", file=sys.stderr)

    print(json.dumps(output))
    return 0 if output["confident"] else 1


def build_online_output(payload: dict, result: SolveResult, session, offset: int) -> dict:
    data = payload.get("data", payload)
    extra = {k: v for k, v in data.items() if k not in IMAGE_FIELDS}
    return {
        "point_x": result.point_x + offset,
        "confident": result.confident,
        "edge_score": result.edge_score,
        "point_y": int(data["pointY"]),
        "cookies": session.cookies.get_dict(),
        "cookie_header": "; ".join(f"{k}={v}" for k, v in session.cookies.get_dict().items()),
        "captcha": extra,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Slider puzzle captcha solver")
    sub = parser.add_subparsers(dest="mode", required=True)

    offline = sub.add_parser("offline", help="solve a saved get-captcha JSON response")
    offline.add_argument("file")
    offline.add_argument("--debug-image")

    online = sub.add_parser("online", help="fetch and solve, print point_x and session data as JSON")
    online.add_argument("--get-url", required=True)
    online.add_argument("--get-method", default="GET")
    online.add_argument("--offset", type=int, default=0)
    online.add_argument("--attempts", type=int, default=3)
    online.add_argument("--cookie")
    online.add_argument("--user-agent", default="Mozilla/5.0")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_offline(args) if args.mode == "offline" else run_online(args)


if __name__ == "__main__":
    sys.exit(main())
