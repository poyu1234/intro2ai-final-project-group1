import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict, Optional

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate SSIM between two images"""
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Ensure same dimensions
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    return ssim(img1, img2, data_range=255)

def find_matching_files(original_folder: str, noisy_folder: str, denoised_folder: str) -> List[Dict[str, Optional[str]]]:
    """Find matching original, noisy, and denoised files."""
    if not os.path.exists(original_folder):
        print(f"Original folder not found: {original_folder}")
        return []
    if not os.path.exists(noisy_folder):
        print(f"Noisy folder not found: {noisy_folder}")
        # Continue, as noisy files might not be part of every evaluation
    if not os.path.exists(denoised_folder):
        print(f"Denoised folder not found: {denoised_folder}")
        # Continue, as denoised files might not be part of every evaluation
    
    original_files = os.listdir(original_folder)
    noisy_files_set = set(os.listdir(noisy_folder)) if os.path.exists(noisy_folder) else set()
    denoised_files_set = set(os.listdir(denoised_folder)) if os.path.exists(denoised_folder) else set()
    
    matching_files_info = []
    
    for original_file_name in original_files:
        if original_file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
            base_name = os.path.splitext(original_file_name)[0]
            
            current_paths: Dict[str, Optional[str]] = {
                'original_path': os.path.join(original_folder, original_file_name),
                'noisy_path': None,
                'denoised_path': None
            }
            
            if base_name.startswith("original_"):
                try:
                    index = base_name.split("_")[1]
                    
                    # Expected noisy file name: noisy_{i}.jpg
                    expected_noisy_name = f"noisy_{index}.jpg"
                    if expected_noisy_name in noisy_files_set:
                        current_paths['noisy_path'] = os.path.join(noisy_folder, expected_noisy_name)
                    else:
                        print(f"Noisy file not found for {original_file_name} (expected: {expected_noisy_name} in {noisy_folder})")
                        
                    # Expected denoised file name: noisy_{i}_denoised.jpg
                    expected_denoised_name = f"noisy_{index}_denoised.jpg"
                    if expected_denoised_name in denoised_files_set:
                        current_paths['denoised_path'] = os.path.join(denoised_folder, expected_denoised_name)
                    else:
                        print(f"Denoised file not found for {original_file_name} (expected: {expected_denoised_name} in {denoised_folder})")
                    
                    matching_files_info.append(current_paths)
                except (IndexError, ValueError):
                    print(f"Could not extract index from original filename: {original_file_name}")
            else:
                print(f"Original filename does not match expected format 'original_{{i}}': {original_file_name}")
    
    return matching_files_info

def evaluate_image_pair(path1: str, path2: str) -> Dict:
    """Evaluate a single pair of images, returning SSIM and status."""
    try:
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        
        if img1 is None:
            return {'error': f"Could not load image: {path1}", 'status': 'error', 
                    'file1': os.path.basename(path1), 'file2': os.path.basename(path2)}
        
        if img2 is None:
            return {'error': f"Could not load image: {path2}", 'status': 'error',
                    'file1': os.path.basename(path1), 'file2': os.path.basename(path2)}
        
        ssim_score = calculate_ssim(img1, img2)
        
        return {
            'file1': os.path.basename(path1),
            'file2': os.path.basename(path2),
            'ssim': ssim_score,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'file1': os.path.basename(path1),
            'file2': os.path.basename(path2),
            'error': str(e),
            'status': 'error'
        }

def run_evaluation(original_folder: str = "dataset/original", 
                  noisy_folder: str = "dataset/noisy",
                  denoised_folder: str = "dataset/denoised",
                  output_csv: str = "evaluation_results.csv") -> Dict:
    """Run complete evaluation for original vs. noisy and original vs. denoised images."""
    
    print("Starting SSIM evaluation...")
    print(f"Original folder: {original_folder}")
    print(f"Noisy folder: {noisy_folder}")
    print(f"Denoised folder: {denoised_folder}")
    
    matching_files_info = find_matching_files(original_folder, noisy_folder, denoised_folder)
    
    if not matching_files_info:
        return {
            'status': 'error',
            'message': 'No matching image sets found to process.',
            'total_original_files_found': 0,
            'statistics_original_vs_noisy': {},
            'statistics_original_vs_denoised': {},
            'results_data': [],
            'output_file': output_csv
        }
    
    print(f"Found {len(matching_files_info)} original images to process.")
    
    results = []
    
    for file_paths_dict in matching_files_info:
        original_path = file_paths_dict['original_path']
        noisy_path = file_paths_dict['noisy_path']
        denoised_path = file_paths_dict['denoised_path']

        if not original_path: # Should not happen if find_matching_files works correctly
            continue

        print(f"Processing: {os.path.basename(original_path)}")
        current_eval_result: Dict[str, Optional[str | float]] = {
            'original_file': os.path.basename(original_path),
            'noisy_file': None, 'ssim_original_vs_noisy': None, 'error_original_vs_noisy': None,
            'denoised_file': None, 'ssim_original_vs_denoised': None, 'error_original_vs_denoised': None
        }

        # Original vs Noisy
        if noisy_path:
            current_eval_result['noisy_file'] = os.path.basename(noisy_path)
            eval_on = evaluate_image_pair(original_path, noisy_path)
            if eval_on['status'] == 'success':
                current_eval_result['ssim_original_vs_noisy'] = eval_on['ssim']
                print(f"  SSIM (Original vs Noisy): {eval_on['ssim']:.4f}")
            else:
                current_eval_result['error_original_vs_noisy'] = eval_on.get('error', 'Unknown error')
                print(f"  Error (Original vs Noisy): {eval_on.get('error', 'Unknown error')}")
        else:
            print(f"  Skipping Original vs Noisy (noisy file not found for {os.path.basename(original_path)})")

        # Original vs Denoised
        if denoised_path:
            current_eval_result['denoised_file'] = os.path.basename(denoised_path)
            eval_od = evaluate_image_pair(original_path, denoised_path)
            if eval_od['status'] == 'success':
                current_eval_result['ssim_original_vs_denoised'] = eval_od['ssim']
                print(f"  SSIM (Original vs Denoised): {eval_od['ssim']:.4f}")
            else:
                current_eval_result['error_original_vs_denoised'] = eval_od.get('error', 'Unknown error')
                print(f"  Error (Original vs Denoised): {eval_od.get('error', 'Unknown error')}")
        else:
            print(f"  Skipping Original vs Denoised (denoised file not found for {os.path.basename(original_path)})")
            
        results.append(current_eval_result)
    
    # Calculate statistics
    stats_on = {}
    ssim_on_scores = [r['ssim_original_vs_noisy'] for r in results if r['ssim_original_vs_noisy'] is not None]
    if ssim_on_scores:
        stats_on = {
            'mean_ssim': np.mean(ssim_on_scores), 'std_ssim': np.std(ssim_on_scores),
            'min_ssim': np.min(ssim_on_scores), 'max_ssim': np.max(ssim_on_scores),
            'count': len(ssim_on_scores)
        }

    stats_od = {}
    ssim_od_scores = [r['ssim_original_vs_denoised'] for r in results if r['ssim_original_vs_denoised'] is not None]
    if ssim_od_scores:
        stats_od = {
            'mean_ssim': np.mean(ssim_od_scores), 'std_ssim': np.std(ssim_od_scores),
            'min_ssim': np.min(ssim_od_scores), 'max_ssim': np.max(ssim_od_scores),
            'count': len(ssim_od_scores)
        }
    
    # Compute improved mean percentage (Noisy → Denoised)
    if stats_on and stats_od and stats_on.get('mean_ssim', 0) > 0:
        improved_mean_pct = (
            (stats_od['mean_ssim'] - stats_on['mean_ssim'])
            / stats_on['mean_ssim'] * 100
        )
    else:
        improved_mean_pct = None

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total original images processed: {len(matching_files_info)}")
    
    if stats_on:
        print(f"\nSSIM Statistics (Original vs. Noisy):")
        print(f"  Evaluated pairs: {stats_on['count']}")
        print(f"  Mean: {stats_on['mean_ssim']:.4f} ± {stats_on['std_ssim']:.4f}")
        print(f"  Range: [{stats_on['min_ssim']:.4f}, {stats_on['max_ssim']:.4f}]")
    else:
        print("\nNo successful (Original vs. Noisy) evaluations.")

    if stats_od:
        print(f"\nSSIM Statistics (Original vs. Denoised):")
        print(f"  Evaluated pairs: {stats_od['count']}")
        print(f"  Mean: {stats_od['mean_ssim']:.4f} ± {stats_od['std_ssim']:.4f}")
        print(f"  Range: [{stats_od['min_ssim']:.4f}, {stats_od['max_ssim']:.4f}]")
    else:
        print("\nNo successful (Original vs. Denoised) evaluations.")
    
    # New: print mean improvement percentage
    if improved_mean_pct is not None:
        print(f"\nMean improvement percentage (Noisy → Denoised): {improved_mean_pct:.2f}%")

    return {
        'status': 'completed' if results else 'no_data_processed',
        'total_original_files_found': len(matching_files_info),
        'statistics_original_vs_noisy': stats_on,
        'statistics_original_vs_denoised': stats_od,
        'mean_improvement_percentage': improved_mean_pct,
        'results_data': results,
        'output_file': output_csv
    }

def plot_results(csv_file: str = "evaluation_results.csv", 
                output_plot: str = "evaluation_plot.png"):
    """Create visualization plots for the evaluation results."""
    try:
        df = pd.read_csv(csv_file)
        
        if df.empty:
            print("No data to plot from CSV.")
            return
        
        # Drop rows where ssim scores are NaN for plotting
        ssim_on = df['ssim_original_vs_noisy'].dropna()
        ssim_od = df['ssim_original_vs_denoised'].dropna()

        if ssim_on.empty and ssim_od.empty:
            print("No valid SSIM scores found in CSV to plot.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if not ssim_on.empty:
            ax.hist(ssim_on, bins=20, alpha=0.7, color='orangered', edgecolor='black', label='Original vs. Noisy')
        if not ssim_od.empty:
            ax.hist(ssim_od, bins=20, alpha=0.7, color='dodgerblue', edgecolor='black', label='Original vs. Denoised')
        
        ax.set_xlabel('SSIM Score')
        ax.set_ylabel('Frequency')
        ax.set_title('SSIM Score Distributions')
        if not ssim_on.empty or not ssim_od.empty:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        # plt.show() # Commented out for non-interactive environments
        print(f"Plot saved to: {output_plot}")
        
    except FileNotFoundError:
        print(f"Error plotting: CSV file not found at {csv_file}")
    except KeyError as e:
        print(f"Error plotting: Column missing in CSV - {e}. Ensure 'ssim_original_vs_noisy' and 'ssim_original_vs_denoised' exist.")
    except Exception as e:
        print(f"Error creating plots: {e}")

def main():
    """Main function to run the evaluation"""
    eval_summary = run_evaluation()
    
    if eval_summary['status'] == 'completed':
        # Check if there are any statistics to plot
        has_noisy_stats = eval_summary['statistics_original_vs_noisy'] and eval_summary['statistics_original_vs_noisy'].get('count', 0) > 0
        has_denoised_stats = eval_summary['statistics_original_vs_denoised'] and eval_summary['statistics_original_vs_denoised'].get('count', 0) > 0
        
        if has_noisy_stats or has_denoised_stats:
            plot_results(csv_file=eval_summary['output_file'])
        else:
            print("Plotting skipped: No successful SSIM evaluations to plot.")
    else:
        print(f"Evaluation did not complete successfully ({eval_summary.get('message', 'Unknown issue')}). Plotting skipped.")
    
    return eval_summary

if __name__ == "__main__":
    main()
