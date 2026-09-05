import numpy as np

class EventDecoder:
    """
    Converts frame-level probabilities into discrete temporal events.
    """
    def __init__(self, threshold=0.5, min_duration=0.1):
        self.threshold = threshold
        self.min_duration = min_duration # Minimum length of an event in seconds
        
        # Reverse mapping from index back to category name
        self.idx_to_class = {
            0: "animal", 1: "vehicle_traffic", 2: "baby_child",
            3: "singing_music", 4: "phone_signal_alarm", 
            5: "appliance_machine", 6: "human_non_speech"
        }

    def decode(self, frame_probs, window_duration=5.0):
        """
        frame_probs: numpy array of shape [Time, Classes]
        window_duration: total duration of the window in seconds
        returns: list of dictionaries {'onset': float, 'offset': float, 'class': str}
        """
        num_frames, num_classes = frame_probs.shape
        frame_sec = window_duration / num_frames
        
        events = []
        
        for c in range(num_classes):
            # 1. Apply threshold to get binary sequence
            binary_seq = (frame_probs[:, c] > self.threshold).astype(int)
            
            # 2. Find transitions (0 to 1 is an onset, 1 to 0 is an offset)
            diff = np.diff(np.pad(binary_seq, (1, 1), mode='constant'))
            onsets = np.where(diff == 1)[0]
            offsets = np.where(diff == -1)[0]
            
            # 3. Convert frame indices to timestamps
            for onset_idx, offset_idx in zip(onsets, offsets):
                onset_time = onset_idx * frame_sec
                offset_time = offset_idx * frame_sec
                
                # 4. Filter out very short artifacts[cite: 1]
                if (offset_time - onset_time) >= self.min_duration:
                    events.append({
                        'onset': round(onset_time, 3),
                        'offset': round(offset_time, 3),
                        'class': self.idx_to_class.get(c, "unknown")
                    })
                    
        return events