package sugarcube.convnet;

public class CNNUtil
{
  private static boolean VOK = false;
  private static double VC = 0.0;

  public static double GaussRandom()
  {
    if (VOK)
    {
      VOK = false;
      return VC;
    }
    double u = 2 * Math.random() - 1;
    double v = 2 * Math.random() - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1)
      return GaussRandom();
    double c = Math.sqrt(-2 * Math.log(r) / r);
    VC = v * c; // cache this
    VOK = true;
    return u * c;
  }

  public static double RandF(double a, double b)
  {
    return Math.random() * (b - a) + a;
  }

  public static double RandI(double a, double b)
  {
    return Math.floor(RandF(a, b));
  }

  public static double RandN(double mu, double std)
  {
    return mu + GaussRandom() * std;
  }

  public static double Max(double[] data)
  {
    double max = data[0];
    for (int i = 1; i < data.length; i++)
      if (data[i] > max)
        max = data[i];
    return max;
  }

  public static int MaxIndex(double[] data)
  {
    int index = 0;
    double max = data[0];
    for (int i = 1; i < data.length; i++)
      if (data[i] > max)
        max = data[index = i];
    return index;
  }

}
