class recognitionModel:
    
    
    
  def __init__(self, user, sample):
      self.user = user
      self.sample = sample
      
  def get_x_train(self):
    # This function reads the time series(x) of the user (Training Sample)
    train_samples = self.user['trainingSamples']
    x = (train_samples[self.sample]['emg'])
    # Divide to 128 for having a signal between -1 and 1
    df = pd.DataFrame.from_dict(x) / 128
    # Apply filter
    train_filtered_X = df.apply(preProcessEMGSegment)
    # Segment the filtered EMG signal
    train_segment_X = EMG_segment(train_filtered_X)
    
    return train_segment_X

    

def preProcessEMGSegment


if __name__ == "__main__":
  user = recognitionModel(user1, 6)
    pass

  