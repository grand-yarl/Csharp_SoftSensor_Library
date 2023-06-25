using CenterSpace.NMath.Core;
using System;
using System.Linq;

namespace LinearRegression
{
    ///<summary>
    /// OrdinaryLinearRegression is class, used for calculating linear model to predict output values
    ///</summary>
    public class OrdinaryLinearRegression
    {
        private DoubleVector? RegressionCoefficients = null;
        private double? Bias = null;
        private int? number_of_singulars;

        /// <summary>
        /// Basic constructor for OrdinaryLinearRegression
        /// </summary>
        public OrdinaryLinearRegression() : this(null) { }
        
        /// <summary>
        /// Constructor, that takes number of singular values
        /// </summary>
        /// <param name="number_of_singular_values">Number of singular values, used to build linear model</param>
        public OrdinaryLinearRegression(in int? number_of_singular_values)
        {
            this.number_of_singulars = number_of_singular_values;
        }
        
        /// <summary>
        /// Method for calculating linear coeffitients of the model
        /// </summary>
        /// <param name="X">Matrix X of input parameters</param>
        /// <param name="Y">Vector Y of output values</param>
        public void fit_model(in double[,] X, in double[] Y)
        {
            DoubleMatrix X_train = new DoubleMatrix(X);
            DoubleVector Y_train = new DoubleVector(Y);

            this.RegressionCoefficients = new DoubleVector(X_train.Cols + 1);

            DoubleMatrix X_new = new DoubleMatrix(X_train.Rows, X_train.Cols + 1, 1, 0);
            for (int i = 0; i < X_train.Rows; i++)
            {
                for (int j = 0; j < X_train.Cols; j++)
                {
                    X_new[i, j] = X_train[i, j];
                }
            }

            DoubleMatrix X_pseudo_inverse = new DoubleMatrix(X_train.Rows, X_train.Cols + 1);
            X_pseudo_inverse = pseudo_inverse(X_new);

            DoubleVector AllCofficients = NMathFunctions.Product(X_pseudo_inverse, Y_train);
            Slice WithoutLastElement = new Slice(0, AllCofficients.Length - 1);

            this.RegressionCoefficients = AllCofficients[WithoutLastElement];
            this.Bias = AllCofficients[AllCofficients.Length - 1];

            //Function for calculating pseudo inverse matrix
            DoubleMatrix pseudo_inverse(DoubleMatrix M)
            {
                var svds = new DoubleSVDecompServer();
                svds.ComputeFull = true;

                var svd = svds.GetDecomp(M);
                DoubleMatrix U = svd.LeftVectors;
                DoubleMatrix V = svd.RightVectors;
                DoubleVector s = svd.SingularValues;
                int number_to_inverse;
                
                if ((this.number_of_singulars is null) || (this.number_of_singulars > svd.Rank))
                    number_to_inverse = svd.Rank;
                else 
                    number_to_inverse = (int)this.number_of_singulars;

                DoubleMatrix Sigma_inv = new DoubleMatrix(M.Cols, M.Rows, 0, 0);
                for (int i = 0;i < number_to_inverse; i++)
                {
                    if (s[i] <= 0.1e-5) break;
                    Sigma_inv[i, i] = 1 / s[i];
                }
                return NMathFunctions.Product(NMathFunctions.Product(V, Sigma_inv), U, ProductTransposeOption.TransposeSecond);
            }
        }

        /// <summary>
        /// Method, that returns regression coefficients if they are calculated
        /// </summary>
        /// <returns>Regression coefficients for input parameters</returns>
        public double[]? getRegressionCoefficients
        {
            get
            {
                if (this.RegressionCoefficients is null) 
                { 
                    Console.WriteLine("The model is not trained! Train model before getting coefficients");
                    return null;
                }
                else
                {
                    double[] DoubleRegressionCoefficients = RegressionCoefficients.ToArray();
                    return DoubleRegressionCoefficients;
                }
            }
        }

        /// <summary>
        /// Method, that returns bias value of linear model
        /// </summary>
        /// <returns>Bias value</returns>
        public double? getBias
        {
            get
            {
                if (this.Bias is null)
                {
                    Console.WriteLine("The model is not trained! Train model before getting coefficients");
                    return null;
                }
                else
                {
                    return this.Bias;
                }
            }
        }

        /// <summary>
        /// Method, that uses linear coefficients to predict new values for input parameters
        /// </summary>
        /// <param name="X">Matrix X of input parameters for prediction</param>
        /// <returns>Vector Y of predicted values</returns>
        public double[]? predict(in double[,] X)
        {
            if ((this.RegressionCoefficients is null) || (this.Bias is null))
            {
                Console.WriteLine("The model is not trained! Train model before getting coefficients");
                return null;
            }
            DoubleMatrix X_test = new DoubleMatrix(X);
            DoubleVector BiasVector = new DoubleVector(X_test.Rows, (double)this.Bias, 0);
            return (NMathFunctions.Product(X_test, this.RegressionCoefficients) + BiasVector).ToArray();
        }
    }
}

