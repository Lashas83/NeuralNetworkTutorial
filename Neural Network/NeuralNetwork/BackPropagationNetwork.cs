using System;
using System.Xml;

namespace NeuralNetwork
{
    public class BackPropagationNetwork
    {
        private int LayerCount { get; set; }

        private int InputSize { get; set; }

        private int[] LayerSize { get; set; }

        private TransferFunction[] TransferFunctions { get; set; }

        private double[][] LayerOutput { get; set; }

        private double[][] LayerInput { get; set; }

        private double[][] Bias { get; set; }

        private double[][] Delta { get; set; }

        private double[][] PreviousBiasDelta { get; set; }

        private double[][][] Weight { get; set; }

        private double[][][] PreviousWeightDelta { get; set; }

        public BackPropagationNetwork(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                throw new ArgumentNullException("filePath");

            _loaded = false;

            Load(filePath);

            _loaded = true;
        }

        public BackPropagationNetwork(int[] inputLayerSizes, TransferFunction[] inputTransferFunctions)
        {
            if (inputTransferFunctions.Length != inputLayerSizes.Length)
                throw new ArgumentException("There is not an equal number of layers and transfer functions.");
            if (inputTransferFunctions[0] != TransferFunction.None)
                throw new ArgumentException("The first transfer function must be None");

            LayerCount = inputLayerSizes.Length - 1;
            InputSize = inputLayerSizes[0];
            LayerSize = new int[LayerCount];

            for (int i = 0; i < LayerCount; i++)
                LayerSize[i] = inputLayerSizes[i + 1];

            TransferFunctions = new TransferFunction[LayerCount];

            for (int i = 0; i < LayerCount; i++)
                TransferFunctions[i] = inputTransferFunctions[i + 1];

            Bias = new double[LayerCount][];
            PreviousBiasDelta = new double[LayerCount][];
            Delta = new double[LayerCount][];
            LayerOutput = new double[LayerCount][];
            LayerInput = new double[LayerCount][];
            Weight = new double[LayerCount][][];
            PreviousWeightDelta = new double[LayerCount][][];

            for (int l = 0; l < LayerCount; l++)
            {
                Bias[l] = new double[LayerSize[l]];
                PreviousBiasDelta[l] = new double[LayerSize[l]];
                Delta[l] = new double[LayerSize[l]];
                LayerOutput[l] = new double[LayerSize[l]];
                LayerInput[l] = new double[LayerSize[l]];

                Weight[l] = new double[l == 0 ? InputSize : LayerSize[l - 1]][];
                PreviousWeightDelta[l] = new double[l == 0 ? InputSize : LayerSize[l - 1]][];

                for (int i = 0; i < (l == 0 ? InputSize : LayerSize[l - 1]); i++)
                {
                    Weight[l][i] = new double[LayerSize[l]];
                    PreviousWeightDelta[l][i] = new double[LayerSize[l]];
                }
            }

            // Initialize the weights
            for (int l = 0; l < LayerCount; l++)
            {
                for (int j = 0; j < LayerSize[l]; j++)
                {
                    Bias[l][j] = Gaussian.GetRandomGaussian();
                    PreviousBiasDelta[l][j] = 0;
                    LayerOutput[l][j] = 0;
                    LayerInput[l][j] = 0;
                    Delta[l][j] = 0;
                }

                for (int i = 0; i < (l == 0 ? InputSize : LayerSize[l - 1]); i++)
                {
                    for (int j = 0; j < LayerSize[l]; j++)
                    {
                        Weight[l][i][j] = Gaussian.GetRandomGaussian();
                        PreviousWeightDelta[l][i][j] = 0;
                    }
                }
            }
        }

        public void Run(ref double[] inputValues, out double[] output)
        {
            if (inputValues.Length != InputSize)
                throw new ArgumentException("Input Data is not of the correct dimension.");

            output = new double[LayerSize[LayerCount - 1]];

            for (int l = 0; l < LayerCount; l++)
            {
                for (int j = 0; j < LayerSize[l]; j++)
                {
                    double sum = 0.0;

                    for (int i = 0; i < (l == 0 ? InputSize : LayerSize[l - 1]); i++)
                    {
                        sum += Weight[l][i][j] * (l == 0 ? inputValues[i] : LayerOutput[l - 1][i]);
                    }

                    sum += Bias[l][j];
                    LayerInput[l][j] = sum;

                    LayerOutput[l][j] = NeuralNetwork.TransferFunctions.Evaluate(TransferFunctions[l], sum);
                }
            }

            for (int i = 0; i < LayerSize[LayerCount - 1]; i++)
            {
                output[i] = LayerOutput[LayerCount - 1][i];
                //Console.WriteLine("For inputs {0} and {1}, output is {2}", inputValues[0], inputValues[1], output[i]);
            }
        }

        public double Train(ref double[] input, ref double[] desired, double trainingRate, double momentum)
        {
            if (input.Length != InputSize)
                throw new ArgumentException("The input is of the wrong dimension.", "input");
            if (desired.Length != LayerSize[LayerCount - 1])
                throw new ArgumentException("The input is of the wrong dimension.", "desired");

            double error = 0.0;
            var output = new double[LayerSize[LayerCount - 1]];

            Run(ref input, out output);

            // do the back propagation
            for (int l = LayerCount - 1; l >= 0; l--)
            {
                if (l == LayerCount - 1) // in the output layer
                {
                    for (int k = 0; k < LayerSize[l]; k++)
                    {
                        var diff = output[k] - desired[k];

                        error += Math.Pow(diff, 2);

                        Delta[l][k] = diff * NeuralNetwork.TransferFunctions.EvaluateDerivative(TransferFunctions[l], LayerInput[l][k]);
                    }
                }
                else // in a hidden layer
                {
                    for (int i = 0; i < LayerSize[l]; i++)
                    {
                        var sum = 0.0;
                        for (int j = 0; j < LayerSize[l + 1]; j++)
                            sum += Weight[l + 1][i][j] * Delta[i][j];

                        Delta[l][i] = sum * NeuralNetwork.TransferFunctions.EvaluateDerivative(TransferFunctions[l], LayerInput[l][i]); ;
                    }
                }
            }

            // update the weights and biases

            for (int l = 0; l < LayerCount; l++)
                for (int i = 0; i < (l == 0 ? InputSize : LayerSize[l - 1]); i++)
                {
                    for (int j = 0; j < LayerSize[l]; j++)
                    {
                        var weightDelta = trainingRate * Delta[l][j] * (l == 0 ? input[i] : LayerOutput[l - 1][i]) + momentum * PreviousWeightDelta[l][i][j];
                        Weight[l][i][j] -= weightDelta;

                        PreviousWeightDelta[l][i][j] = weightDelta;
                    }
                }

            for (int l = 0; l < LayerCount; l++)
            {
                for (int i = 0; i < LayerSize[l]; i++)
                {
                    var biasDelta = trainingRate * Delta[l][i];
                    Bias[l][i] -= biasDelta + momentum * PreviousBiasDelta[l][i];

                    PreviousBiasDelta[l][i] = biasDelta;
                }
            }

            return error;
        }

        public void Save(string FilePath)
        {
            if (string.IsNullOrEmpty(FilePath))
                throw new ArgumentNullException("FilePath");

            var writer = XmlWriter.Create(FilePath);

            writer.WriteStartElement("NeuralNetwork");
            writer.WriteAttributeString("Type", "BackPropagation");

            // Parameters element
            writer.WriteStartElement("Parameters");

            writer.WriteElementString("Name", Name);
            writer.WriteElementString("InputSize", InputSize.ToString());
            writer.WriteElementString("LayerCount", LayerCount.ToString());

            // Layer sizes
            writer.WriteStartElement("Layers");

            for (int l = 0; l < LayerCount; l++)
            {
                writer.WriteStartElement("Layer");

                writer.WriteAttributeString("Index", l.ToString());
                writer.WriteAttributeString("Size", LayerSize[l].ToString());
                writer.WriteAttributeString("Type", TransferFunctions[l].ToString());

                writer.WriteEndElement(); // Layer
            }

            writer.WriteEndElement(); // Layers

            writer.WriteEndElement(); // Parameters

            writer.WriteStartElement("Weights");

            for (int l = 0; l < LayerCount; l++)
            {
                writer.WriteStartElement("Layer");

                writer.WriteAttributeString("Index", l.ToString());
                for (int j = 0; j < LayerSize[l]; j++)
                {
                    writer.WriteStartElement("Node");

                    writer.WriteAttributeString("Index", j.ToString());
                    writer.WriteAttributeString("Bias", Bias[l][j].ToString());

                    for (int i = 0; i < (l == 0 ? InputSize : LayerSize[l - 1]); i++)
                    {
                        writer.WriteStartElement("Axon");

                        writer.WriteAttributeString("Index", i.ToString());
                        writer.WriteString(Weight[l][i][j].ToString());

                        writer.WriteEndElement(); // Axon
                    }

                    writer.WriteEndElement(); // Node
                }

                writer.WriteEndElement(); // Layer
            }

            writer.WriteEndElement(); // Weights

            writer.WriteEndElement(); // NeuralNetwork

            writer.Flush();
            writer.Close();
        }

        public void Load(string FilePath)
        {
            if (string.IsNullOrEmpty(FilePath))
                throw new ArgumentNullException("FilePath");

            _doc = new XmlDocument();
            _doc.Load(FilePath);

            string basePath = "";
            string nodePath = "";
            double value;

            // Load from xml

            if (XPathValue("NeuralNetwork/@Type") != "BackPropagation")
                return;

            basePath = "NeuralNetwork/Parameters";

            Name = XPathValue(basePath + "/Name");

            int inputSize;
            int.TryParse(XPathValue(basePath + "/InputSize"), out inputSize);
            InputSize = inputSize;

            int layerCount;
            int.TryParse(XPathValue(basePath + "/LayerCount"), out layerCount);
            LayerCount = layerCount;

            LayerSize = new int[layerCount];
            TransferFunctions = new TransferFunction[layerCount];

            basePath += "/Layers/Layer";

            for (int l = 0; l < layerCount; l++)
            {
                int layerSizeOfL;
                int.TryParse(XPathValue(basePath + "[@Index='" + l + "']/@Size"), out layerSizeOfL);
                LayerSize[l] = layerSizeOfL;

                TransferFunction transferFunctionOfL;
                Enum.TryParse(XPathValue(basePath + "[@Index='" + l + "']/@Type"), out transferFunctionOfL);
                TransferFunctions[l] = transferFunctionOfL;
            }

            // Parse Weights element
            for (int l = 0; l < LayerCount; l++)
            {
                basePath = "NeuralNetwork/Weights/Layer[@Index='" + l + "']/";
                for (int j = 0; j < LayerSize[l]; j++)
                {
                    nodePath = "Node[@Index='" + j + "']/@Bias";
                    double biasOfLJ;
                    double.TryParse(XPathValue(basePath + nodePath), out biasOfLJ);
                    Bias[l][j] = biasOfLJ;
                    PreviousBiasDelta[l][j] = 0;
                    LayerOutput[l][j] = 0;
                    LayerInput[l][j] = 0;
                    Delta[l][j] = 0;
                }

                for (int i = 0; i < (l == 0 ? InputSize : LayerSize[l - 1]); i++)
                {
                    for (int j = 0; j < LayerSize[l]; j++)
                    {
                        nodePath = "Node[@Index='" + j + "']/Axon[@Index='" + i + "']";
                        double weightOfLij;
                        double.TryParse(XPathValue(basePath + nodePath), out weightOfLij);
                        Weight[l][i][j] = weightOfLij;
                        PreviousWeightDelta[l][i][j] = 0;
                    }
                }
            }

            // release
            _doc = null;
        }

        private string XPathValue(string xPath)
        {
            if (string.IsNullOrEmpty(xPath))
                throw new ArgumentNullException("xPath");

            XmlNode node = _doc.SelectSingleNode(xPath);

            if (node == null)
            {
                throw new XmlException(string.Format("Cannot find specific node '{0}'", xPath));
            }

            return node.InnerText;
        }

        /// <summary>
        /// A publically accessible string containing the name of the neural network.
        /// </summary>
        public string Name = "Default";

        private XmlDocument _doc = null;

        private bool _loaded = true;
    }
}
