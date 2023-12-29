package sugarcube.convnet.layer;

import sugarcube.convnet.CNN;
import sugarcube.convnet.CNNVolumetricData;
import sugarcube.convnet.CNNVolumetricDataList;

public class CNNLayer
{
  public static final String INPUT = "input";
  public static final String CONV = "conv";
  public static final String POOL = "pool";
  public static final String FC = "fc";
  public static final String LRN = "lrn";
  public static final String DROPOUT = "dropout";
  public static final String SOFTMAX = "softmax";
  public static final String REGRESSION = "regression";
  public static final String MAXOUT = "maxout";
  public static final String SVM = "svm";
  public static final String IDENTITY = null;

  public String type;
  public CNNVolumetricData in;
  public CNNVolumetricData out;
  public String fct;

  public CNNLayer(String type, CNNVolumetricData in)
  {
    this.type = type;
    this.in = in;
  }

  public CNNLayer function(String fct)
  {
    this.fct = fct;
    return this;
  }
  
  public boolean isFct(String fct)
  {
    return this.fct==fct || this.fct!=null && this.fct.equals(fct);
  }

  public boolean isIdentityFct()
  {
    return this.fct == null || this.fct.equals(CNN.IDENTITY);
  }

  public void forward(boolean isTraining)
  {
    double v[] = out.v;
    if (!isIdentityFct())
      switch (fct)
      {
      case CNN.IDENTITY:
        break;
      case CNN.RELU:
        for (int i = 0; i < out.volume; i++)
          if (v[i] < 0)
            v[i] = 0;
        break;
      case CNN.SIGMOID:
        for (int i = 0; i < out.volume; i++)
          v[i] = 1.0 / (1.0 + Math.exp(-v[i]));
        break;
      case CNN.TANH:
        for (int i = 0; i < out.volume; i++)
          v[i] = Math.tanh(v[i]);
        break;
      case CNN.DUDA:
        break;
      }
  }

  public void backward()
  {
    double v[] = out.v;
    double dv[] = out.dv;
    if (!isIdentityFct())
      switch (fct)
      {
      case CNN.IDENTITY:
        break;
      case CNN.RELU:
        for (int i = 0; i < out.volume; i++)
          if (v[i] <= 0)
            dv[i] = 0;
        break;
      case CNN.SIGMOID:
        for (int i = 0; i < out.volume; i++)
          dv[i] = v[i] * (1.0 - v[i]) * dv[i];
        break;
      case CNN.TANH:
        for (int i = 0; i < out.volume; i++)
          dv[i] = (1.0 - v[i] * v[i]) * dv[i];
        break;
      case CNN.DUDA:
        break;
      }
  }

  public double loss()
  {
    return 0.0;
  }

  public CNNVolumetricDataList weights()
  {
    CNNVolumetricDataList weights = new CNNVolumetricDataList();
    weights.dataList(filters()).dataList(biases());
    return weights;
  }

  public CNNVolumetricData[] filters()
  {
    return new CNNVolumetricData[0];
  }

  public CNNVolumetricData biases()
  {
    return null;
  }

  public boolean hasWeights()
  {
    return filters().length > 0;
  }

  public double l1DecayMultiply()
  {
    return 0.0;
  }

  public double l2DecayMultiply()
  {
    return 0.0;
  }

}
