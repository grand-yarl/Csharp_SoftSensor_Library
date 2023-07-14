using CenterSpace.NMath.Core;
using System;
using System.Linq;
using System.Xml.Linq;

namespace LinearRegression
{
    ///<summary>
    /// OrdinaryLinearRegression is class, used for calculating linear model to predict output values
    ///</summary>
    public class OrdinaryLinearRegression
    {
        private protected DoubleVector? RegressionCoefficients = null;
        private protected double? Bias = null;
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
        public virtual void Fit_model(in double[,] X, in double[] Y)
        {
            DoubleMatrix X_train = new DoubleMatrix(X);
            DoubleVector Y_train = new DoubleVector(Y);

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

            DoubleVector AllCoefficients = NMathFunctions.Product(X_pseudo_inverse, Y_train);
            Slice WithoutLastElement = new Slice(0, AllCoefficients.Length - 1);

            this.RegressionCoefficients = AllCoefficients[WithoutLastElement];
            this.Bias = AllCoefficients[AllCoefficients.Length - 1];

            //Function for calculating pseudo inverse matrix
            DoubleMatrix pseudo_inverse(in DoubleMatrix M)
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
        public double[]? GetRegressionCoefficients
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
        public double? GetBias
        {
            get
            {
                if (this.Bias is null)
                {
                    Console.WriteLine("The model is not trained! Train model before getting coefficients using - OrdinaryLinearRegression.Fit_model(double[,] X, double[] Y)");
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
        public double[]? Predict(in double[,] X)
        {
            if ((this.RegressionCoefficients is null) || (this.Bias is null))
            {
                Console.WriteLine("The model is not trained! Train model before getting coefficients using - OrdinaryLinearRegression.Fit_model(double[,] X, double[] Y)");
                return null;
            }
            DoubleMatrix X_test = new DoubleMatrix(X);
            DoubleVector BiasVector = new DoubleVector(X_test.Rows, (double)this.Bias, 0);
            return (NMathFunctions.Product(X_test, this.RegressionCoefficients) + BiasVector).ToArray();
        }
    }


    public class GradientLinearRegression : OrdinaryLinearRegression
    {
        private double tolerance = 0.1;
        private double gradient_rate = 0.001;
        private double delta = 0.001;

        public GradientLinearRegression() : base() { }

        public GradientLinearRegression(double tolerance)
        {
            this.tolerance = tolerance;
        }

        public GradientLinearRegression(double tolerance, double gradient_rate) : this(tolerance)
        {
            this.gradient_rate = gradient_rate;
        }

        public GradientLinearRegression(double tolerance, double gradient_rate, double delta) : this(tolerance, gradient_rate)
        {
            this.delta = delta;
        }

        public override void Fit_model(in double[,] X, in double[] Y)
        {
            DoubleMatrix X_train = new DoubleMatrix(X);
            DoubleVector Y_train = new DoubleVector(Y);

            DoubleVector AllCoefficients = new DoubleVector(X_train.Cols + 1, 0, 0);

            DoubleVector gradient_vector;

            for (ulong i = 0; i < 10e100; i++)
            {
                gradient_vector = gradient(X_train, Y_train, AllCoefficients);
                
                if (NMathFunctions.AbsSum(gradient_vector) < this.tolerance)
                {
                    break;
                }
                AllCoefficients = AllCoefficients - this.gradient_rate*gradient_vector;
            }

            Slice WithoutLastElement = new Slice(0, AllCoefficients.Length - 1);

            this.RegressionCoefficients = AllCoefficients[WithoutLastElement];
            this.Bias = AllCoefficients[AllCoefficients.Length - 1];



            DoubleVector gradient(in DoubleMatrix X, in DoubleVector Y, in DoubleVector Coefficients)
            {
                DoubleVector gradient_vector = new DoubleVector(X_train.Cols + 1);
                DoubleVector plusdeltaCoefficients = new DoubleVector(Coefficients);
                DoubleVector minusdeltaCoefficients = new DoubleVector(Coefficients);
                for (int i = 0; i < Coefficients.Length; i++)
                {
                    
                    plusdeltaCoefficients[i] += this.delta;
                    if (i > 0)
                    {
                        plusdeltaCoefficients[i - 1] -= this.delta;
                    }
                    minusdeltaCoefficients[i] -= this.delta;
                    if (i > 0)
                    {
                        minusdeltaCoefficients[i - 1] += this.delta;
                    }

                    gradient_vector[i] = (square_error(X, Y, plusdeltaCoefficients) - square_error(X, Y, minusdeltaCoefficients)) / this.delta;

                }
                
                return gradient_vector;
            }

            double square_error(in DoubleMatrix X, in DoubleVector Y, in DoubleVector AllCoefficients)
            {
                Slice WithoutLastElement = new Slice(0, AllCoefficients.Length - 1);

                DoubleVector regressioncoefficients = AllCoefficients[WithoutLastElement];
                double bias = AllCoefficients[AllCoefficients.Length - 1];
                DoubleVector BiasVector = new DoubleVector(X.Rows, bias, 0);

                DoubleVector full_error = (NMathFunctions.Product(X, regressioncoefficients) + BiasVector) - Y;

                return NMathFunctions.Dot(full_error, full_error);
        }

        }

    }
}

namespace ComponentAnalysis
{
    /// <summary>
    /// PCA - Principal components analysis ia a class, used to calculate new orthogonal basis, expaining most of data variances
    /// </summary>
    public class PCA
    {
        private DoubleMatrix? PCA_vectors = null;
        private DoubleVector? PCA_values = null;

        /// <summary>
        /// Basic constructor of PCA
        /// </summary>
        public PCA() { }

        /// <summary>
        /// Method, that is used to calculate principal components and loadings
        /// </summary>
        /// <param name="X">Analized data matrix X</param>
        public void Analize(double[,] X)
        {
            DoubleMatrix X_analized = new DoubleMatrix(X);
            double x_row_mean = 0;

            for (int i = 0; i < X_analized.Cols; i++)
            {
                for (int j = 0; j < X_analized.Rows; j++)
                {
                    x_row_mean += X_analized[j, i];
                }
                x_row_mean /= X_analized.Rows;
                for (int j = 0; j < X_analized.Rows; j++)
                {
                    X_analized[j, i] -= x_row_mean;
                }
                x_row_mean = 0;
            }

            DoubleMatrix XX = NMathFunctions.Product(X_analized.Transpose(), X_analized);

            DoubleSymEigDecomp eigen_decomposition = new DoubleSymEigDecomp(XX);

            int size = eigen_decomposition.EigenValues.Length;

            this.PCA_values = new DoubleVector(size);
            this.PCA_vectors = new DoubleMatrix(eigen_decomposition.EigenVectors.Rows, eigen_decomposition.EigenVectors.Cols);

            for (int i = 0; i < size; i++) 
            {
                this.PCA_values[i] = eigen_decomposition.EigenValues[size - i - 1];
                for (int j = 0; j < eigen_decomposition.EigenVectors.Cols; j++)
                {
                    this.PCA_vectors[i, j] = eigen_decomposition.EigenVectors[j, size - i - 1];
                }
            }
        }

        /// <summary>
        /// Method, that returns singular values of analized data, sorted from biggest to smallest
        /// </summary>
        /// <param name="number_of_singulars">Number of singular values</param>
        /// <returns>Singular values of analized data</returns>
        public double[]? GetSingularValues(int number_of_singulars = -1)
        {
            if (this.PCA_values is null)
            {
                Console.WriteLine("The PCA model is not analized! Analize data before getting singular values using - PCA.Analize(double[,] X)");
                return null;
            }
            else
            {
                if ((number_of_singulars < 0) || (number_of_singulars > PCA_values.Length))
                {
                    number_of_singulars = this.PCA_values.Length;
                }
                
                Slice singular_value_slice = new Slice(0, number_of_singulars);
                double[] pca_values = this.PCA_values[singular_value_slice].ToArray();
                for (int i = 0; i < number_of_singulars; i++)
                {
                    pca_values[i] = Math.Sqrt(pca_values[i]);
                }
                return pca_values;
            }
        }

        /// <summary>
        /// Method, that returns the persentage of explained variance by first principal components
        /// </summary>
        /// <param name="number_of_components">Number of principal components</param>
        /// <returns>Persentage of explained variance</returns>
        public double? ExplainedVarianceRatio(int number_of_components = -1)
        {
            if (this.PCA_values is null)
            {
                Console.WriteLine("The PCA model is not analized! Analize data before getting singular values using - PCA.Analize(double[,] X)");
                return null;
            }
            else
            {
                if ((number_of_components < 0) || (number_of_components > PCA_values.Length))
                {
                    number_of_components = this.PCA_values.Length;
                }

                Slice singular_value_slice = new Slice(0, number_of_components);
                double total = this.PCA_values.Sum();
                double explained = this.PCA_values[singular_value_slice].Sum();
                return explained / total * 100;
            }
        }

        /// <summary>
        /// Method, that returns the persentages, explained by each principal component
        /// </summary>
        /// <param name="number_of_components">Number of principal components</param>
        /// <returns>Persentages of explained variance for each component</returns>
        public double[]? EveryVectorVarianceRatio(int number_of_components = -1)
        {
            if (this.PCA_values is null)
            {
                Console.WriteLine("The PCA model is not analized! Analize data before getting principal components using - PCA.Analize(double[,] X)");
                return null;
            }
            else
            {
                if ((number_of_components < 0) || (number_of_components > PCA_values.Length))
                {
                    number_of_components = this.PCA_values.Length;
                }

                Slice singular_value_slice = new Slice(0, number_of_components);
                double total = this.PCA_values.Sum();
                double[] vector_explaination = new double[number_of_components];
                for (int i = 0; i < number_of_components; i++)
                {
                    vector_explaination[i] = this.PCA_values[i] / total;
                }
                return vector_explaination;
            }
        }

        /// <summary>
        /// Method, that returns first principal components, sorted by their importance
        /// </summary>
        /// <param name="number_of_components">Number of principal components</param>
        /// <returns>Principal components</returns>
        public double[,]? GetPrincipalComponents(int number_of_components = -1)
        {
            if ((this.PCA_vectors is null) || (this.PCA_values is null))
            {
                Console.WriteLine("The PCA model is not analized! Analize data before getting singular values using - PCA.Analize(double[,] X)");
                return null;
            }
            else
            {
                if ((number_of_components < 0) || (number_of_components > PCA_values.Length))
                {
                    number_of_components = this.PCA_values.Length;
                }

                Slice rows_slice = new Slice(0, number_of_components);
                Slice cols_slice = new Slice(0, this.PCA_vectors.Cols);
                return this.PCA_vectors[rows_slice, cols_slice].ToArray();
            }
        }
    }
}

