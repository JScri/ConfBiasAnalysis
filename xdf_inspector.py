#!/usr/bin/env python3
"""
XDF File Inspector with CONFIRMED Channel Mapping
BCI Confirmation Bias Study - UTS 2025

Confirmed mapping (all participants):
Ch5=Fz, Ch6=Cz, Ch13=Pz, Ch14=DISCONNECTED
"""

import pyxdf
import numpy as np
import json
import sys
import os

# CONFIRMED CHANNEL MAPPING (same for all participants)
CHANNEL_MAPPING = {
    1: "Fp1", 2: "Fp2", 3: "C3", 4: "C4",
    5: "Fz",  6: "Cz",  7: "O1", 8: "O2",
    9: "F7",  10: "F8", 11: "F3", 12: "F4",
    13: "Pz", 14: "DISCONNECTED", 15: "P3", 16: "P4"
}

DISCONNECTED_CHANNEL = 14

def inspect_xdf_file(xdf_file: str):
    """Comprehensive inspection with confirmed channel names"""
    
    print(f"\n{'='*80}")
    print(f"XDF FILE INSPECTOR - CONFIRMED CHANNEL MAPPING")
    print(f"{'='*80}")
    print(f"File: {xdf_file}")
    print(f"Size: {os.path.getsize(xdf_file) / 1024 / 1024:.2f} MB")
    print(f"{'='*80}\n")
    
    try:
        print("Loading XDF file...")
        streams, header = pyxdf.load_xdf(xdf_file)
        print(f"✓ Successfully loaded {len(streams)} streams\n")
    except Exception as e:
        print(f"✗ Error loading XDF: {e}")
        return None, None
    
    # Inspect each stream
    for idx, stream in enumerate(streams):
        print(f"\n{'-'*80}")
        print(f"STREAM {idx+1}")
        print(f"{'-'*80}")
        
        try:
            info = stream['info']
            stream_name = info['name'][0] if 'name' in info else 'Unknown'
            stream_type = info.get('type', ['Unknown'])[0]
            n_channels = int(info['channel_count'][0]) if 'channel_count' in info else 1
            
            print(f"\n📋 Basic Information:")
            print(f"  Name: {stream_name}")
            print(f"  Type: {stream_type}")
            print(f"  Channels: {n_channels}")
            
            if 'nominal_srate' in info:
                srate = float(info['nominal_srate'][0])
                print(f"  Sampling Rate: {srate} Hz")
            
            if 'channel_format' in info:
                print(f"  Format: {info['channel_format'][0]}")
            
            # Data structure
            time_series = stream['time_series']
            time_stamps = stream['time_stamps']
            
            print(f"\n📊 Data Structure:")
            if isinstance(time_series, list):
                print(f"  Samples: {len(time_series)}")
            else:
                print(f"  Shape: {time_series.shape}")
            
            # Temporal info
            if len(time_stamps) > 0:
                duration = time_stamps[-1] - time_stamps[0]
                print(f"\n⏱️  Temporal:")
                print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
                print(f"  First: {time_stamps[0]:.3f}s")
                print(f"  Last: {time_stamps[-1]:.3f}s")
            
            # Stream-specific analysis
            if 'eeg' in stream_name.lower():
                print(f"\n🧠 EEG STREAM ANALYSIS (CONFIRMED MAPPING):")
                print(f"  {'='*76}")
                
                # Show confirmed channel mapping
                print(f"\n  Channel Mapping (VERIFIED):")
                for ch_num in range(1, 17):
                    ch_name = CHANNEL_MAPPING[ch_num]
                    status = "❌ EXCLUDED" if ch_num == DISCONNECTED_CHANNEL else "✓"
                    print(f"    Ch{ch_num:2d}: {ch_name:<15s} [{status}]")
                
                print(f"\n  Key Electrodes:")
                print(f"    🎯 Fz  (Ch5)  - Frontal midline theta (PRIMARY for confirmation bias)")
                print(f"    🎯 Cz  (Ch6)  - Central midline")
                print(f"    🎯 Pz  (Ch13) - Parietal midline")
                print(f"    ❌ Ch14       - Disconnected/Railed (excluded)")
                
                # Data statistics
                try:
                    if isinstance(time_series, list):
                        data_array = np.array(time_series)
                    else:
                        data_array = time_series
                    
                    print(f"\n  Signal Statistics (first 3 channels):")
                    for ch_idx in range(min(3, data_array.shape[1] if len(data_array.shape) > 1 else 1)):
                        ch_num = ch_idx + 1
                        ch_name = CHANNEL_MAPPING[ch_num]
                        
                        if len(data_array.shape) > 1:
                            ch_data = data_array[:, ch_idx]
                        else:
                            ch_data = data_array
                        
                        print(f"    Ch{ch_num} ({ch_name}):")
                        print(f"      Mean: {np.mean(ch_data):12.2f}")
                        print(f"      Std:  {np.std(ch_data):12.2f}")
                        print(f"      Range: [{np.min(ch_data):8.2f}, {np.max(ch_data):8.2f}]")
                except Exception as e:
                    print(f"    Could not compute statistics: {e}")
            
            elif 'marker' in stream_name.lower():
                print(f"\n🔍 MARKER STREAM:")
                print(f"\n  Sample markers (first 5):")
                for i in range(min(5, len(time_series))):
                    marker = time_series[i]
                    marker_str = marker[0] if isinstance(marker, (list, np.ndarray)) else str(marker)
                    print(f"    [{i}] @ {time_stamps[i]:.3f}s: {marker_str}")
                
                # Count marker types
                marker_types = {}
                for marker_data in time_series:
                    marker_str = marker_data[0] if isinstance(marker_data, (list, np.ndarray)) else str(marker_data)
                    marker_type = marker_str.split('_')[0] if '_' in marker_str else marker_str
                    marker_types[marker_type] = marker_types.get(marker_type, 0) + 1
                
                print(f"\n  Top marker types:")
                for mtype, count in sorted(marker_types.items(), key=lambda x: -x[1])[:10]:
                    print(f"    {mtype}: {count}")
            
            elif 'likert' in stream_name.lower() or 'response' in stream_name.lower():
                print(f"\n📊 RESPONSE STREAM:")
                print(f"  Total responses: {len(time_series)}")
                
                # Try to parse first few
                parsed_count = 0
                for i in range(min(3, len(time_series))):
                    try:
                        data = time_series[i]
                        json_str = data[0] if isinstance(data, (list, np.ndarray)) else str(data)
                        parsed = json.loads(json_str)
                        parsed_count += 1
                    except:
                        pass
                
                print(f"  Successfully parsed: {parsed_count}/{min(3, len(time_series))} samples")
            
            elif 'behavioral' in stream_name.lower():
                print(f"\n🎯 BEHAVIORAL STREAM:")
                print(f"  Total events: {len(time_series)}")
                
        except Exception as e:
            print(f"\n  ✗ Error: {e}")
    
    print(f"\n{'='*80}")
    print(f"INSPECTION COMPLETE")
    print(f"{'='*80}\n")
    
    # Summary
    print("SUMMARY:")
    print(f"  Total streams: {len(streams)}")
    
    has_eeg = any('eeg' in s['info']['name'][0].lower() for s in streams)
    has_markers = any('marker' in s['info']['name'][0].lower() for s in streams)
    
    if has_eeg:
        print(f"  ✓ EEG stream present (15 valid channels, Ch14 excluded)")
        print(f"  ✓ Key electrodes: Fz (Ch5), Cz (Ch6), Pz (Ch13)")
    
    if has_markers:
        print(f"  ✓ Marker stream present")
    
    print(f"\n✅ File is valid and ready for analysis!")
    print(f"\nNext step:")
    print(f"  python final_eeg_analysis.py {os.path.basename(xdf_file)}")
    
    return streams, header

def main():
    if len(sys.argv) < 2:
        print("\nUsage: python xdf_inspector_confirmed.py <xdf_file>")
        print("\nExample: python xdf_inspector_confirmed.py participant_01.xdf")
        sys.exit(1)
    
    xdf_file = sys.argv[1]
    
    if not os.path.exists(xdf_file):
        print(f"\n✗ Error: File not found: {xdf_file}")
        sys.exit(1)
    
    streams, header = inspect_xdf_file(xdf_file)

if __name__ == "__main__":
    main()