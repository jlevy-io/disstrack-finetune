"""
tools/compare_results.py

Compare v1 vs v2 model results and generate HTML report
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import Counter
import statistics


def load_results(file_path: Path) -> Dict:
    """Load results JSON file"""
    with open(file_path) as f:
        return json.load(f)


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate metrics from results"""
    all_roasts = []
    all_times = []
    unique_words = set()

    for result in results:
        if result["success"]:
            all_times.append(result["inference_time_seconds"])
            for roast in result["candidates"]:
                all_roasts.append(roast)
                unique_words.update(roast.lower().split())

    lengths = [len(r) for r in all_roasts]

    return {
        "total_roasts": len(all_roasts),
        "avg_length": statistics.mean(lengths) if lengths else 0,
        "median_length": statistics.median(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "std_length": statistics.stdev(lengths) if len(lengths) > 1 else 0,
        "avg_time": statistics.mean(all_times) if all_times else 0,
        "median_time": statistics.median(all_times) if all_times else 0,
        "vocabulary_size": len(unique_words),
        "unique_words": sorted(list(unique_words))[:100]  # First 100 for display
    }


def generate_html_report(
    v1_data: Dict,
    v2_data: Dict,
    v1_metrics: Dict,
    v2_metrics: Dict,
    output_path: Path
):
    """Generate HTML comparison report"""

    # Match up results by image_id
    v1_by_id = {r["image_id"]: r for r in v1_data["results"] if r["success"]}
    v2_by_id = {r["image_id"]: r for r in v2_data["results"] if r["success"]}
    common_ids = set(v1_by_id.keys()) & set(v2_by_id.keys())

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>v1 vs v2 Model Comparison</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 2rem;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .subtitle {{
            color: #94a3b8;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}
        
        .model-card {{
            background: #1e293b;
            border-radius: 12px;
            padding: 2rem;
            border: 2px solid;
        }}
        
        .model-card.v1 {{
            border-color: #3b82f6;
        }}
        
        .model-card.v2 {{
            border-color: #10b981;
        }}
        
        .model-card h2 {{
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .model-card.v1 h2 {{
            color: #3b82f6;
        }}
        
        .model-card.v2 h2 {{
            color: #10b981;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }}
        
        .metric {{
            background: #0f172a;
            padding: 1rem;
            border-radius: 8px;
        }}
        
        .metric-label {{
            color: #94a3b8;
            font-size: 0.875rem;
            margin-bottom: 0.25rem;
        }}
        
        .metric-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #e2e8f0;
        }}
        
        .metric-unit {{
            font-size: 0.875rem;
            color: #64748b;
            margin-left: 0.25rem;
        }}
        
        .comparison-section {{
            background: #1e293b;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid #334155;
        }}
        
        .comparison-header {{
            display: flex;
            align-items: center;
            gap: 1.5rem;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #334155;
        }}
        
        .comparison-header img {{
            width: 200px;
            height: 200px;
            object-fit: cover;
            border-radius: 8px;
            border: 2px solid #334155;
        }}
        
        .image-info h3 {{
            font-size: 1.3rem;
            margin-bottom: 0.5rem;
        }}
        
        .ground-truth {{
            background: #0f172a;
            padding: 1rem;
            border-radius: 8px;
            border-left: 3px solid #f59e0b;
            margin-top: 0.5rem;
        }}
        
        .ground-truth-label {{
            color: #f59e0b;
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        
        .ground-truth-text {{
            color: #cbd5e1;
        }}
        
        .roasts-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }}
        
        .roast-column h4 {{
            font-size: 1.2rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid;
        }}
        
        .roast-column.v1 h4 {{
            color: #3b82f6;
            border-color: #3b82f6;
        }}
        
        .roast-column.v2 h4 {{
            color: #10b981;
            border-color: #10b981;
        }}
        
        .roast-item {{
            background: #0f172a;
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-radius: 8px;
            border-left: 3px solid;
            transition: all 0.2s;
        }}
        
        .roast-item:hover {{
            transform: translateX(4px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }}
        
        .roast-column.v1 .roast-item {{
            border-color: #3b82f6;
        }}
        
        .roast-column.v2 .roast-item {{
            border-color: #10b981;
        }}
        
        .roast-text {{
            color: #e2e8f0;
            line-height: 1.5;
            margin-bottom: 0.5rem;
        }}
        
        .roast-meta {{
            font-size: 0.85rem;
            color: #64748b;
        }}
        
        .winner-badge {{
            display: inline-block;
            background: #f59e0b;
            color: #0f172a;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-left: 0.5rem;
        }}
        
        @media (max-width: 1024px) {{
            .roasts-comparison {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 Model Comparison: v1 vs v2</h1>
        <p class="subtitle">
            {len(common_ids)} images compared • Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}
        </p>
        
        <div class="summary">
            <div class="model-card v1">
                <h2>📊 v1 (Two-Stage Training)</h2>
                <div class="metric-grid">
                    <div class="metric">
                        <div class="metric-label">Avg Length</div>
                        <div class="metric-value">
                            {v1_metrics['avg_length']:.1f}
                            <span class="metric-unit">chars</span>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Length Range</div>
                        <div class="metric-value">
                            {v1_metrics['min_length']}-{v1_metrics['max_length']}
                            <span class="metric-unit">chars</span>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Avg Time</div>
                        <div class="metric-value">
                            {v1_metrics['avg_time']:.2f}
                            <span class="metric-unit">sec</span>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Vocabulary</div>
                        <div class="metric-value">
                            {v1_metrics['vocabulary_size']:,}
                            <span class="metric-unit">words</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="model-card v2">
                <h2>📊 v2 (Simple Training)</h2>
                <div class="metric-grid">
                    <div class="metric">
                        <div class="metric-label">Avg Length</div>
                        <div class="metric-value">
                            {v2_metrics['avg_length']:.1f}
                            <span class="metric-unit">chars</span>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Length Range</div>
                        <div class="metric-value">
                            {v2_metrics['min_length']}-{v2_metrics['max_length']}
                            <span class="metric-unit">chars</span>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Avg Time</div>
                        <div class="metric-value">
                            {v2_metrics['avg_time']:.2f}
                            <span class="metric-unit">sec</span>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Vocabulary</div>
                        <div class="metric-value">
                            {v2_metrics['vocabulary_size']:,}
                            <span class="metric-unit">words</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
"""

    # Add comparisons
    for image_id in sorted(common_ids):
        v1_result = v1_by_id[image_id]
        v2_result = v2_by_id[image_id]

        # Determine winner by average length (closer to target of ~71 chars)
        v1_avg_len = sum(len(r) for r in v1_result["candidates"]) / len(v1_result["candidates"])
        v2_avg_len = sum(len(r) for r in v2_result["candidates"]) / len(v2_result["candidates"])
        target = 71
        v1_diff = abs(v1_avg_len - target)
        v2_diff = abs(v2_avg_len - target)
        v1_winner = v1_diff < v2_diff

        image_rel_path = f"../data/raw/images/{v1_result['image_filename']}"

        html += f"""
        <div class="comparison-section">
            <div class="comparison-header">
                <img src="{image_rel_path}" alt="{v1_result['image_filename']}">
                <div class="image-info">
                    <h3>{v1_result['image_filename']}</h3>
                    <div class="ground-truth">
                        <div class="ground-truth-label">Ground Truth Roast:</div>
                        <div class="ground-truth-text">{v1_result['ground_truth']}</div>
                    </div>
                </div>
            </div>
            
            <div class="roasts-comparison">
                <div class="roast-column v1">
                    <h4>
                        v1 (Two-Stage)
                        {'<span class="winner-badge">CLOSER TO TARGET</span>' if v1_winner else ''}
                    </h4>
"""
        for i, roast in enumerate(v1_result["candidates"], 1):
            html += f"""
                    <div class="roast-item">
                        <div class="roast-text">{roast}</div>
                        <div class="roast-meta">#{i} • {len(roast)} chars</div>
                    </div>
"""
        html += f"""
                </div>
                
                <div class="roast-column v2">
                    <h4>
                        v2 (Simple)
                        {'<span class="winner-badge">CLOSER TO TARGET</span>' if not v1_winner else ''}
                    </h4>
"""
        for i, roast in enumerate(v2_result["candidates"], 1):
            html += f"""
                    <div class="roast-item">
                        <div class="roast-text">{roast}</div>
                        <div class="roast-meta">#{i} • {len(roast)} chars</div>
                    </div>
"""
        html += """
                </div>
            </div>
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def print_summary(v1_data: Dict, v2_data: Dict, v1_metrics: Dict, v2_metrics: Dict):
    """Print terminal summary"""
    print(f"\n{'='*70}")
    print(f"📊 COMPARISON SUMMARY")
    print("=" * 70)
    print()

    print(f"v1 (Two-Stage):")
    print(f"  Roasts generated: {v1_metrics['total_roasts']}")
    print(f"  Avg length: {v1_metrics['avg_length']:.1f} chars (target: ~71)")
    print(f"  Length range: {v1_metrics['min_length']}-{v1_metrics['max_length']} chars")
    print(f"  Std deviation: {v1_metrics['std_length']:.1f} chars")
    print(f"  Avg time: {v1_metrics['avg_time']:.2f}s per image")
    print(f"  Vocabulary: {v1_metrics['vocabulary_size']:,} unique words\n")

    print(f"v2 (Simple):")
    print(f"  Roasts generated: {v2_metrics['total_roasts']}")
    print(f"  Avg length: {v2_metrics['avg_length']:.1f} chars (target: ~71)")
    print(f"  Length range: {v2_metrics['min_length']}-{v2_metrics['max_length']} chars")
    print(f"  Std deviation: {v2_metrics['std_length']:.1f} chars")
    print(f"  Avg time: {v2_metrics['avg_time']:.2f}s per image")
    print(f"  Vocabulary: {v2_metrics['vocabulary_size']:,} unique words\n")

    # Comparison
    print("📈 Key Differences:")
    
    length_diff = v2_metrics['avg_length'] - v1_metrics['avg_length']
    length_emoji = "📈" if length_diff > 0 else "📉"
    print(f"  {length_emoji} Length: v2 is {abs(length_diff):.1f} chars {'longer' if length_diff > 0 else 'shorter'}")

    time_diff = v2_metrics['avg_time'] - v1_metrics['avg_time']
    time_emoji = "⚡" if time_diff < 0 else "🐌"
    print(f"  {time_emoji} Speed: v2 is {abs(time_diff):.2f}s {'faster' if time_diff < 0 else 'slower'}")

    vocab_diff = v2_metrics['vocabulary_size'] - v1_metrics['vocabulary_size']
    vocab_emoji = "📚" if vocab_diff > 0 else "📖"
    print(f"  {vocab_emoji} Vocabulary: v2 uses {abs(vocab_diff)} {'more' if vocab_diff > 0 else 'fewer'} unique words")

    std_diff = v2_metrics['std_length'] - v1_metrics['std_length']
    consistency_emoji = "🎯" if std_diff < 0 else "📊"
    print(f"  {consistency_emoji} Consistency: v2 is {'more' if std_diff < 0 else 'less'} consistent ({abs(std_diff):.1f} chars)")

    print(f"\n{'='*70}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare v1 and v2 model results")
    parser.add_argument("--v1-results", required=True, help="Path to v1 results JSON")
    parser.add_argument("--v2-results", required=True, help="Path to v2 results JSON")
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Output directory for report"
    )

    args = parser.parse_args()

    # Load data
    print(f"\n{'='*70}")
    print(f"🔥 DissTrack Model Comparison")
    print(f"{'='*70}\n")

    print(f"Loading v1 results: {args.v1_results}")
    v1_data = load_results(Path(args.v1_results))

    print(f"Loading v2 results: {args.v2_results}")
    v2_data = load_results(Path(args.v2_results))

    # Calculate metrics
    print("\n📊 Calculating metrics...")
    v1_metrics = calculate_metrics(v1_data["results"])
    v2_metrics = calculate_metrics(v2_data["results"])

    # Generate report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_file = output_dir / f"comparison_report_{timestamp}.html"

    print(f"📝 Generating HTML report...")
    generate_html_report(v1_data, v2_data, v1_metrics, v2_metrics, html_file)

    # Print summary
    print_summary(v1_data, v2_data, v1_metrics, v2_metrics)

    print(f"✅ Report generated: {html_file}")
    print(f"\n💡 Open the report in your browser to see the visual comparison!\n")


if __name__ == "__main__":
    main()
