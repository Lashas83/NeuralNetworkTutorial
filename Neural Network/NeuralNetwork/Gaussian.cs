using System;

namespace NeuralNetwork
{
    public static class Gaussian
    {
        private static readonly Random Gen = new Random();

        public static double GetRandomGaussian()
        {
            return GetRandomGaussian(0, 1);
        }

        public static double GetRandomGaussian(double mean, double stdDev)
        {
            double rVal1;
            double rVal2;

            GetRandomGaussian(mean, stdDev, out rVal1, out rVal2);

            return rVal1;
        }

        public static void GetRandomGaussian(double mean, double stdDev, out double val1, out double val2)
        {
            double u;
            double v;
            double s;
            double t;

            do
            {
                u = 2 * Gen.NextDouble() - 1;
                v = 2 * Gen.NextDouble() - 1;
            } while (u * u + v * v > 1 || (u == 0 && v == 0));

            s = u * u + v * v;
            t = Math.Sqrt((-2.0 * Math.Log(s)) / s);

            val1 = stdDev * u * t + mean;
            val2 = stdDev * v * t + mean;
        }

    }
}