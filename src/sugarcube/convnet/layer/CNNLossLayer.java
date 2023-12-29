package sugarcube.convnet.layer;

import sugarcube.convnet.CNNVolumetricData;
import sugarcube.convnet.CNNUtil;

// Layers that implement a loss. Currently these are the layers that 
// can initiate a backward() pass. In future we probably want a more 
// flexible system that can accomodate multiple losses to do multi-task
// learning, and stuff like that. But for now, one of the layers in this
// file must be the final layer in a Net.

// This is a classifier, with N discrete classes from 0 to N-1
// it gets a stream of N incoming numbers and computes the softmax
// function (exponentiate and normalize to sum to 1 as probabilities should)
public class CNNLossLayer extends CNNLayer
{
  public double loss;
  public int groundtruthIndex;

  public CNNLossLayer(String type, CNNVolumetricData in)
  {
    super(type, in);
    this.out = new CNNVolumetricData(1,1,in.volume, 0.0);
  }
  
  @Override
  public double loss()
  {
    return loss;
  }
  
  public boolean maxIsGroundtruth()
  {
    return maxIndex() == groundtruthIndex;
  }
  
  public double max()
  {
    return CNNUtil.Max(out.v);
  }
  
  public int maxIndex()
  {
    return CNNUtil.MaxIndex(out.v);
  }
 

}
