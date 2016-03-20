using NeuralNetwork;

namespace NeuralNetworkTutorialApp
{
    using System;
    using System.Diagnostics;

    class Program
    {
        private const string _filePath = @"test_network.xml";

        static void Main(string[] args)
        {
            var layerSizes = new[] { 2, 2, 1 };

            var transferFunctions = new[]
            {TransferFunction.None, TransferFunction.Sigmoid, TransferFunction.Linear};

            var backPropagationNetwork = new BackPropagationNetwork(layerSizes, transferFunctions)
            {
                Name = "XOR-Gate Example"
            };

            var input = new[]
            {
                new[] {0.0, 0.0},
                new [] {1.0, 0.0},
                new [] {0.0, 1.0},
                new [] {1.0, 1.0},
            };

            var expected = new[]
            {
                new[] {0.0},
                new[] {1.0},
                new[] {1.0},
                new[] {0.0}
            };


            double error = 0.0;
            const int maxCount = 100000;
            int count = 0;

            Stopwatch watch = Stopwatch.StartNew();

            do
            {
                // prepare for training epic
                count++;
                error = 0;

                // train
                for (int i = 0; i < input.Length; i++)
                {
                    error += backPropagationNetwork.Train(ref input[i], ref expected[i], .15, .1);
                }

                if (count % 100 == 0 || error <= 0.0001 || count > maxCount)
                {
                    Console.WriteLine("Epoch {0} completed with error {1:0.0000}", count, error);
                }
            } while (error > 0.0001 && count <= maxCount);

            watch.Stop();

            Console.WriteLine("Training complete in {0}", watch.Elapsed);

            var output = new double[4][];

            for (int i = 0; i < input.Length; i++)
                backPropagationNetwork.Run(ref input[i], out output[i]);

            for (int i = 0; i < input.Length; i++)
                Console.WriteLine("For inputs {0} and {1}, output is {2}", input[i][0], input[i][1], output[i][0]);

            Console.WriteLine("Time Elapsed :" + watch.Elapsed);
            Console.WriteLine("Hit Enter...");
            Console.ReadLine();
        }
    }
}
