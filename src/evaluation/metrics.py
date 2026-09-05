import numpy as np

def calculate_event_f1(pred_events, ref_events):
    """
    Calculates Event-based F1 with +/- 20% tolerance (minimum 50ms).
    """
    tp, fp, fn = 0, 0, len(ref_events)
    matched_refs = set()
    
    for p_event in pred_events:
        p_onset = p_event['onset']
        p_offset = p_event['offset']
        match_found = False
        
        for r_idx, r_event in enumerate(ref_events):
            if r_idx in matched_refs:
                continue
                
            r_onset = float(r_event['start'])
            r_offset = float(r_event['end'])
            r_duration = r_offset - r_onset
            
            tol = max(0.20 * r_duration, 0.05)
            
            if abs(p_onset - r_onset) <= tol and abs(p_offset - r_offset) <= tol:
                tp += 1
                fn -= 1
                matched_refs.add(r_idx)
                match_found = True
                break
                
        if not match_found:
            fp += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, precision, recall

def calculate_segment_dice(pred_frames, ref_frames):
    """
    Computes temporal overlap using the Segment Dice Coefficient[cite: 1].
    Expected input: binary numpy arrays of 10ms frame predictions.
    """
    intersection = np.sum(pred_frames * ref_frames)
    total_area = np.sum(pred_frames) + np.sum(ref_frames)
    
    if total_area == 0:
        return 1.0 # Perfect match if both are empty
        
    return (2.0 * intersection) / total_area

def evaluate_predictions(predictions_dict, ground_truth_dict):
    """
    Returns Event F1, Segment Dice, and the Combined Score[cite: 1].
    """
    f1_scores = []
    dice_scores = []
    
    for clip_id in predictions_dict:
        preds = predictions_dict[clip_id].get('events', [])
        refs = ground_truth_dict[clip_id].get('events', [])
        
        f1, _, _ = calculate_event_f1(preds, refs)
        f1_scores.append(f1)
        
        # Compute pseudo-frames for Dice assuming 10ms hop
        # This requires converting the event dicts back to 10ms boolean arrays
        max_time = max([p['offset'] for p in preds] + [float(r['end']) for r in refs] + [0])
        total_frames = int(np.ceil(max_time / 0.01))
        
        p_mask = np.zeros(total_frames)
        r_mask = np.zeros(total_frames)
        
        for p in preds:
            p_mask[int(p['onset']/0.01):int(p['offset']/0.01)] = 1
        for r in refs:
            r_mask[int(float(r['start'])/0.01):int(float(r['end'])/0.01)] = 1
            
        dice_scores.append(calculate_segment_dice(p_mask, r_mask))
        
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    
    return {
        "Event_F1": avg_f1,
        "Segment_Dice": avg_dice,
        "Combined": avg_f1 + avg_dice
    }