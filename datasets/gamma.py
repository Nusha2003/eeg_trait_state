import pandas as pd
import os

# Set the root directory to start the search
root_dir = './results' 
# Define the output filename for your master table
output_file = 'master_hierarchy_summary.csv'

def process_all_hierarchy_files(start_path):
    all_data = []

    for root, dirs, files in os.walk(start_path):
        if os.path.basename(root).lower() == 'hierarchy':
            for file in files:
                if file.endswith('.csv'):
                    full_path = os.path.abspath(os.path.join(root, file))
                    
                    try:
                        df = pd.read_csv(full_path)
                        
                        # Calculate stats
                        stats = df.groupby('space')['hier_ratio'].agg(['mean', 'std']).reset_index()
                        
                        # Add identifying metadata
                        stats['full_file_path'] = full_path
                        stats['file_name'] = file
                        
                        all_data.append(stats)
                        
                    except Exception as e:
                        print(f"Error processing file at {full_path}: {e}")

    if all_data:
        # 1. Combine all individual results into one DataFrame
        final_report = pd.concat(all_data, ignore_index=True)
        
        # 2. Reorder columns for a logical "Master Table" look
        column_order = ['full_file_path', 'file_name', 'space', 'mean', 'std']
        final_report = final_report[column_order]
        
        # 3. Save to CSV
        final_report.to_csv(output_file, index=False)
        print(f"Successfully saved master table to: {os.path.abspath(output_file)}")
        
        return final_report
    else:
        return None

# Execute
summary_df = process_all_hierarchy_files(root_dir)

if summary_df is not None:
    print("\n--- Master Table Preview ---")
    pd.set_option('display.max_colwidth', None)
    print(summary_df.head(10).to_string(index=False))