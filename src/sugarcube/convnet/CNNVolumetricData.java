package sugarcube.convnet;


import java.awt.image.BufferedImage;
import java.util.Arrays;

public class CNNVolumetricData
{
    public int width;
    public int height;
    public int depth;
    public int volume;
    public double[] v;// rgbrgbrgbrgb
    public double[] dv;
    public boolean isBias = false;
    // tmp data only used while training
    public transient double[] gSum;
    public transient double[] xSum;

    public CNNVolumetricData(int width, int height, int depth, double init)
    {
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.volume = width * height * depth;
        this.v = new double[volume];
        this.dv = new double[volume];
        if (Double.isNaN(init))
        {
            // weight normalization is done to equalize the output
            // variance of every neuron, otherwise neurons with a lot
            // of incoming connections have outputs of larger variance
            double scale = Math.sqrt(1.0 / volume);
            for (int i = 0; i < volume; i++)
                this.v[i] = CNNUtil.RandN(0.0, scale);
        } else if (init != 0.0)
            Arrays.fill(v, init);
    }

    public CNNVolumetricData bias()
    {
        this.isBias = true;
        return this;
    }

    public float[] floatValues()
    {
        float[] values = new float[v.length];
        for (int i = 0; i < values.length; i++)
            values[i] = (float) v[i];
        return values;
    }

    public CNNVolumetricData toIdentity()
    {
        Arrays.fill(v, 0.0);
        v[v.length / 2] = 1;
        return this;
    }

    public CNNVolumetricData training(boolean on)
    {
        this.gSum = on ? new double[v.length] : null;
        this.xSum = on ? new double[v.length] : null;
        return this;
    }

    public void reset(double v0)
    {
        Arrays.fill(v, v0);
    }

    public int i(int x, int y, int z)
    {
        return ((width * y) + x) * depth + z;
    }

    public double value(int x, int y, int z)
    {
        return v[i(x, y, z)];
    }

    public void setValue(int x, int y, int z, double v)
    {
        this.v[i(x, y, z)] = v;
    }

    public void setValues(double... values)
    {
        this.v = values;
    }

    public void setValues(BufferedImage image)
    {
        int h = image.getHeight();
        int w = image.getWidth();
        int i = 0;

        int[] p = new int[image.getColorModel().getNumColorComponents()];
        stop:
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                image.getRaster().getPixel(x, y, p);
                for (int b = 0; b < p.length; b++)
                {
                    if (i < v.length)
                        this.v[i++] = p[b] / 255.0;
                    else
                    {
                        System.out.println("CNNCube.setValues - break at " + i);
                        break stop;
                    }
                }
            }
    }

    public void addValue(int x, int y, int z, double v)
    {
        this.v[i(x, y, z)] += v;
    }

    public void resetValues()
    {
        Arrays.fill(v, 0.0);
    }

    public double grad(int x, int y, int z)
    {
        return dv[i(x, y, z)];
    }

    public void setGrad(int x, int y, int z, double dv)
    {
        this.dv[i(x, y, z)] = dv;
    }

    public void addGrad(int x, int y, int z, double dv)
    {
        this.dv[i(x, y, z)] += dv;
    }

    public void resetGrad()
    {
        Arrays.fill(dv, 0.0);
    }

    public void add(CNNVolumetricData vol)
    {
        for (int i = 0; i < v.length; i++)
            v[i] += vol.v[i];
    }

    public void add(CNNVolumetricData vol, double scale)
    {
        for (int i = 0; i < v.length; i++)
            v[i] += vol.v[i] * scale;
    }

    public String arrayString()
    {
        StringBuilder sb = new StringBuilder(v.length * 4);
        for (int i = 0; i < v.length; i++)
        {
            String str = ((int) (100 * v[i])) + "";
            while (str.length() < 4)
                str = " " + str;
            sb.append(str);
        }
        return sb.toString();
    }

    public String dArrayString()
    {
        StringBuilder sb = new StringBuilder(dv.length * 4);
        for (int i = 0; i < dv.length; i++)
        {
            String str = ((int) (100 * dv[i])) + "";
            while (str.length() < 4)
                str = " " + str;
            sb.append(str);
        }
        return sb.toString();
    }

    public String cubeString()
    {
        StringBuilder sb = new StringBuilder(v.length * 4);
        for (int z = 0; z < depth; z++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    String str = ((int) (100 * this.value(x, y, z))) + "";
                    while (str.length() < 4)
                        str = " " + str;
                    sb.append(str);
                }
                sb.append("\n");
            }
            sb.append("\n\n");
        }
        return sb.toString();
    }

    public CNNVolumetricData newZero()
    {
        return new CNNVolumetricData(width, height, depth, 0.0);
    }

    public CNNVolumetricData copy()
    {
        CNNVolumetricData vol = newZero();
        System.arraycopy(v, 0, vol.v, 0, v.length);
        System.arraycopy(dv, 0, vol.dv, 0, dv.length);
        return vol;
    }
}
