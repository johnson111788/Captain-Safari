"""
Merge specific columns from a CSV file

Usage:
    python utils/merge_csv.py \
        --base_csv ./data/metadata.encoded.csv \
        --source_csv ./data/metadata.encoded_with_image.csv \
        --columns image_y_latent,image_single_latent \
        --output_csv ./data/metadata.encoded.csv
"""

import pandas as pd
import argparse
import os

def merge_csv_columns(base_csv_path, source_csv_path, columns_to_merge, output_csv_path=None, backup=True):
    """
    Extract specified columns from source_csv and merge them into base_csv
    
    Args:
        base_csv_path: base CSV file path
        source_csv_path: source CSV file path (source of columns to extract)
        columns_to_merge: list of column names to merge
        output_csv_path: output CSV file path (if None, will overwrite base_csv)
        backup: whether to backup the original file
    """
    print(f"Reading base CSV: {base_csv_path}")
    base_df = pd.read_csv(base_csv_path)
    print(f"  - Number of rows: {len(base_df)}")
    print(f"  - Number of columns: {len(base_df.columns)}")
    print(f"  - Existing columns: {list(base_df.columns)}")
    
    print(f"\nReading source CSV: {source_csv_path}")
    source_df = pd.read_csv(source_csv_path)
    print(f"  - Number of rows: {len(source_df)}")
    print(f"  - Number of columns: {len(source_df.columns)}")
    
    # Check if the number of rows is consistent
    if len(base_df) != len(source_df):
        print(f"\nWarning: the number of rows in the two CSV files is inconsistent!")
        print(f"  - Base file: {len(base_df)} rows")
        print(f"  - Source file: {len(source_df)} rows")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled")
            return
    
    # Check if the columns to merge exist
    missing_columns = [col for col in columns_to_merge if col not in source_df.columns]
    if missing_columns:
        print(f"\nError: the source file is missing the following columns: {missing_columns}")
        print(f"Available columns: {list(source_df.columns)}")
        return
    
    # Check if any columns will be overwritten
    existing_columns = [col for col in columns_to_merge if col in base_df.columns]
    if existing_columns:
        print(f"\nWarning: the following columns already exist in the base file, they will be overwritten: {existing_columns}")
    
    # Merge columns
    print(f"\nMerging columns: {columns_to_merge}")
    for col in columns_to_merge:
        base_df[col] = source_df[col]
        print(f"  ✓ Merged column: {col}")
    
    # Determine output path
    if output_csv_path is None:
        output_csv_path = base_csv_path
    
    # Backup original file
    if backup and output_csv_path == base_csv_path:
        backup_path = base_csv_path + '.backup'
        if os.path.exists(backup_path):
            print(f"\nBackup file already exists: {backup_path}")
            response = input("Overwrite backup? (y/n): ")
            if response.lower() != 'y':
                print("Skip backup")
            else:
                print(f"Backing up original file to: {backup_path}")
                base_df_original = pd.read_csv(base_csv_path)
                base_df_original.to_csv(backup_path, index=False)
                print("  ✓ Backup completed")
        else:
            print(f"\nBacking up original file to: {backup_path}")
            base_df_original = pd.read_csv(base_csv_path)
            base_df_original.to_csv(backup_path, index=False)
            print("  ✓ Backup completed")
    
    # Save results
    print(f"\nSaving results to: {output_csv_path}")
    base_df.to_csv(output_csv_path, index=False)
    print("  ✓ Save completed")
    
    print(f"\nMerged CSV information:")
    print(f"  - Number of rows: {len(base_df)}")
    print(f"  - Number of columns: {len(base_df.columns)}")
    print(f"  - All columns: {list(base_df.columns)}")
    
    # Verify merged columns
    print(f"\nVerifying merged columns:")
    for col in columns_to_merge:
        non_null_count = base_df[col].notna().sum()
        print(f"  - {col}: {non_null_count}/{len(base_df)} non-null values")

def main():
    parser = argparse.ArgumentParser(description='Merge specific columns from a CSV file')
    
    parser.add_argument('--base_csv', type=str, required=True, help='Base CSV file path')
    parser.add_argument('--source_csv', type=str, required=True, help='Source CSV file path (source of columns to extract)')
    parser.add_argument('--columns', type=str, required=True, help='List of column names to merge, separated by commas (e.g.: image_y_latent,image_single_latent)')
    parser.add_argument('--output_csv', type=str, default=None, help='Output CSV file path (default: overwrite base file)')
    parser.add_argument('--no_backup', action='store_true', help='Do not backup original file')
    
    args = parser.parse_args()
    
    # Parse column names
    columns_to_merge = [col.strip() for col in args.columns.split(',')]
    
    print("="*60)
    print("CSV column merging tool")
    print("="*60)
    print(f"Base file: {args.base_csv}")
    print(f"Source file: {args.source_csv}")
    print(f"Columns to merge: {columns_to_merge}")
    print(f"Output file: {args.output_csv or args.base_csv}")
    print(f"Backup: {'No' if args.no_backup else 'Yes'}")
    print("="*60)
    print()
    
    # Execute merge
    merge_csv_columns(
        base_csv_path=args.base_csv,
        source_csv_path=args.source_csv,
        columns_to_merge=columns_to_merge,
        output_csv_path=args.output_csv,
        backup=not args.no_backup
    )
    
    print("\n✓ Merge completed!")


if __name__ == '__main__':
    main()

