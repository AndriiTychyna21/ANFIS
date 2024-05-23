import java.util.Random;

public class ANFIS {
    double[] var = new double[3];
    double[][][] func = new double[3][2][2];
    double[][] funcResults = new double[3][2];
    double[] w = new double[2];

    double w_sum;
    double[] w_norm = new double[2];
    double[][] rules = new double[2][3];
    double[] q = new double[2];
    double F;
    double learningRate = 0.1;

    double[][] images = {{1, 32, 3, 3.141362},
            {2, 32, 2, 6.29715},
            {3, 45, 3, 0.316198},
            {4, 65, 4, 2.229654},
            {5, 3, 53, 12.56472},
            {6, 2, 23, 0.973952},
            {7, 1, 12, 1.339259},
            {8, 5, 6, 46.39461},
            {9, 4, 5, 1.551375},
            {5, 3, 7, 12.8396},
            {4, 7, 6, 1.986992},
            {3, 5, 4, 0.673534},
            {2, 2, 11, 5.947558},
            {1, 8, 21, 3.146682},
            {2, 2, 21, 5.64757},
            {6, 11, 76, 0.405182},
            {7, 24, 34, 1.219278},
            {8, 56, 54, 47.27631},
            {9, 75, 23, 1.770304},
            {23, 43, 76, 3.150859},
            {43, 32, 32, 3.245126},
            {54, 43, 38, 0.849992},
            {23, 21, 16, 2.905126},
            {23, 43, 76, 1.380608}
    };

    double[][] test = {{12, 21, 34},
            {14, 14, 14},
            {20,20,20}
    };
    public ANFIS(){
        Random rand = new Random();
        for (int i = 0; i < func.length; i++){
            for (int j = 0; j < func[i].length; j++){
                func[i][j][0] = rand.nextDouble(5, 10);
                func[i][j][1] = rand.nextDouble(20, 30);
            }
        }
        for (int i = 0; i < rules.length; i++){
            for (int j = 0; j < rules[i].length; j++){
                rules[i][j] = rand.nextDouble(0.4, 0.5);
            }
        }
        shuffle();
        normalise();
    }

    private void normalise(){
        for (int i = 0; i < images.length; i++){
            for (int j = 0; j < images[i].length; j++){
                images[i][j] = 1.0/(1 + Math.exp(-images[i][j]));
            }
        }
        for (int i = 0; i < test.length; i++){
            for (int j = 0; j < test[i].length; j++){
                test[i][j] = 1.0/(1 + Math.exp(-test[i][j]));
            }
        }
    }

    private double denormalise(double num){
        double result = -Math.log((1-num)/num);
        return result;
    }

    private void shuffle() {
        int index;
        double[] temp;
        Random random = new Random();
        for (int i = images.length - 1; i > 0; i--) {
            index = random.nextInt(i);
            temp = images[index];
            images[index] = images[i];
            images[i] = temp;
        }
    }
    public double calculate(double x, double y, double z){
        var[0] = x;
        var[1] = y;
        var[2] = z;
        for (int i = 0; i < funcResults.length; i++){
            for (int j = 0; j < funcResults[i].length; j++){
                double number = -Math.pow((var[i] - func[i][j][0])/func[i][j][1], 2);
                funcResults[i][j] = Math.exp(number);
            }
        }
        w_sum = 0;
        for (int i = 0; i < w.length; i++){
            w[i] =  funcResults[0][i] * funcResults[1][i] * funcResults[2][i];
            w_sum += w[i];
        }
        for (int i = 0; i < w_norm.length; i++){
            w_norm[i] = w[i]/w_sum;
        }
        for (int i = 0; i < q.length; i++){
            double sum = 0;
            for (int j = 0; j < var.length; j++){
                sum += rules[i][j] * var[j];
            }
            q[i] = w_norm[i] * sum;
        }
        F = q[0] + q[1];
        return F;
    }

    public void learn(){
        double R, mistake;
        for (int im = 0; im < images.length; im++){
            R = calculate(images[im][0], images[im][1], images[im][2]);
            mistake = Math.pow(R - images[im][3], 2)/2;
            System.out.println("Mistake: " + mistake);
            for (int i = 0; i < func.length; i++){
                for (int j = 0; j < func[i].length; j++){
                    double change1 = 2 * learningRate * (R - images[im][3]) * ((q[j] + w[j] * F)/w_sum) * (var[i]-func[i][j][0])/(func[i][j][1] * func[i][j][1]);
                    double change2 = 2 * learningRate * (R - images[im][3]) * ((q[j] + w[j] * F)/w_sum) * Math.pow(var[i]-func[i][j][0], 2)/Math.pow(func[i][j][1], 3);
                    func[i][j][0] -= change1;
                    func[i][j][1] -= change2;
                }
            }
            for (int i = 0; i < rules.length; i++){
                for (int j = 0; j < rules[i].length; j++){
                    double change = learningRate * (R - images[im][3]) * w_norm[i] * var[j];
                    rules[i][j] -= change;
                }
            }
            System.out.println("Result: " + denormalise(R));
        }
    }

    public void testing(){
        System.out.println();
        for (int i = 0; i < test.length; i ++){
            double R = calculate(test[i][0], test[i][1], test[i][2]);
            System.out.println("control image #" + (i+1) + ": " + denormalise(R));
        }
    }
}
