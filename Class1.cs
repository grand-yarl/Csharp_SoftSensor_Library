using CenterSpace.NMath.Core;
using System;

namespace LinearRegression
{
    public class OrdinaryLinearRegression
    {
        private DoubleVector LinearCoefficients;
        private int? number_of_singulars;

        public OrdinaryLinearRegression() : this(null) { }
        
        public OrdinaryLinearRegression(in int? number_of_singular_values)
        {
            this.number_of_singulars = number_of_singular_values;
        }

        public void fit_model(in DoubleMatrix X_train, in DoubleVector Y_train)
        {
            this.LinearCoefficients = new DoubleVector(X_train.Cols + 1);

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
            

            this.LinearCoefficients = NMathFunctions.Product(X_pseudo_inverse, Y_train);
            Console.WriteLine(this.LinearCoefficients);


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
                Console.WriteLine(Sigma_inv.ToString());
                return NMathFunctions.Product(NMathFunctions.Product(V, Sigma_inv), U, ProductTransposeOption.TransposeSecond);
            }
            
        }

        public DoubleVector predict(in DoubleMatrix X_test)
        {
            Slice WithoutLastElement = new Slice(0, this.LinearCoefficients.Length - 1);
            DoubleVector RegressionCoeffs = LinearCoefficients[WithoutLastElement];
            double FreeCoeff = this.LinearCoefficients[this.LinearCoefficients.Length - 1];
            DoubleVector AddFreeCoeff = new DoubleVector(X_test.Rows, FreeCoeff, 0);
            return NMathFunctions.Product(X_test, RegressionCoeffs) + FreeCoeff;
        }





    }
}

