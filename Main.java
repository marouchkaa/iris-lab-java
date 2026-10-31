package iris_java;

import javax.swing.*;
import java.awt.*;
import java.util.*;

public class Main {

    // ---------- Red & Gray theme ----------
    static final Color RED = new Color(0xC0392B);
    static final Color DARK_RED = new Color(0x7B1E1E);
    static final Color GRAY = new Color(0x7F7F7F);
    static final Color DARK_GRAY = new Color(0x3A3A3A);
    static final Color LIGHT_GRAY = new Color(0xEDEDED);
    static final Color[] PALETTE = {RED, GRAY, DARK_GRAY};

    // ---------- Data ----------
    static final String[] FEATURES = {"Petal length (cm)", "Petal width (cm)"};
    static final String[] SPECIES = {"Setosa", "Versicolor", "Virginica"};

    static double[][] x;   // [n][2]
    static int[] truth;    // true group of each flower

    static void generateData() {
        double[][] mean = {{1.5, 0.25}, {4.3, 1.3}, {5.5, 2.0}};
        double[][] sd = {{0.17, 0.08}, {0.45, 0.20}, {0.55, 0.27}};
        int perGroup = 50;
        Random rnd = new Random(42);
        x = new double[perGroup * 3][2];
        truth = new int[perGroup * 3];
        int i = 0;
        for (int g = 0; g < 3; g++) {
            for (int j = 0; j < perGroup; j++, i++) {
                x[i][0] = Math.max(0.1, mean[g][0] + rnd.nextGaussian() * sd[g][0]);
                x[i][1] = Math.max(0.05, mean[g][1] + rnd.nextGaussian() * sd[g][1]);
                truth[i] = g;
            }
        }
    }

    // ---------- Statistics ----------
    static double mean(double[] a) {
        double s = 0;
        for (double v : a) s += v;
        return s / a.length;
    }

    static double std(double[] a) {
        double m = mean(a), s = 0;
        for (double v : a) s += (v - m) * (v - m);
        return Math.sqrt(s / (a.length - 1));
    }

    static double[] column(int c) {
        double[] col = new double[x.length];
        for (int i = 0; i < x.length; i++) col[i] = x[i][c];
        return col;
    }

    // returns {slope, intercept, r}
    static double[] regression(double[] a, double[] b) {
        double ma = mean(a), mb = mean(b);
        double cov = 0, va = 0, vb = 0;
        for (int i = 0; i < a.length; i++) {
            cov += (a[i] - ma) * (b[i] - mb);
            va += (a[i] - ma) * (a[i] - ma);
            vb += (b[i] - mb) * (b[i] - mb);
        }
        double slope = cov / va;
        return new double[]{slope, mb - slope * ma, cov / Math.sqrt(va * vb)};
    }

    // ---------- K-Means ----------
    static class KMeansResult {
        int[] labels;
        double[][] centroids;
        double inertia;
    }

    static double dist2(double[] p, double[] q) {
        double dx = p[0] - q[0], dy = p[1] - q[1];
        return dx * dx + dy * dy;
    }

    static KMeansResult kmeansOnce(int k, long seed) {
        Random rnd = new Random(seed);
        int n = x.length;
        double[][] c = new double[k][];
        Set<Integer> used = new HashSet<>();
        for (int i = 0; i < k; i++) {
            int idx;
            do { idx = rnd.nextInt(n); } while (!used.add(idx));
            c[i] = x[idx].clone();
        }
        int[] labels = new int[n];
        Arrays.fill(labels, -1);
        for (int iter = 0; iter < 100; iter++) {
            boolean changed = false;
            for (int i = 0; i < n; i++) {
                int best = 0;
                double bd = Double.MAX_VALUE;
                for (int j = 0; j < k; j++) {
                    double d = dist2(x[i], c[j]);
                    if (d < bd) { bd = d; best = j; }
                }
                if (labels[i] != best) { labels[i] = best; changed = true; }
            }
            double[][] sum = new double[k][2];
            int[] count = new int[k];
            for (int i = 0; i < n; i++) {
                sum[labels[i]][0] += x[i][0];
                sum[labels[i]][1] += x[i][1];
                count[labels[i]]++;
            }
            for (int j = 0; j < k; j++) {
                if (count[j] > 0) {
                    c[j][0] = sum[j][0] / count[j];
                    c[j][1] = sum[j][1] / count[j];
                }
            }
            if (!changed) break;
        }
        KMeansResult r = new KMeansResult();
        r.labels = labels;
        r.centroids = c;
        for (int i = 0; i < n; i++) r.inertia += dist2(x[i], c[labels[i]]);
        return r;
    }

    // best of 10 random starts
    static KMeansResult kmeans(int k) {
        KMeansResult best = null;
        for (int s = 0; s < 10; s++) {
            KMeansResult r = kmeansOnce(k, 100 + s);
            if (best == null || r.inertia < best.inertia) best = r;
        }
        return best;
    }

    // share of flowers that belong to the majority true group of their cluster
    static double purity(KMeansResult r, int k) {
        int correct = 0;
        for (int c = 0; c < k; c++) {
            int[] votes = new int[3];
            for (int i = 0; i < x.length; i++) if (r.labels[i] == c) votes[truth[i]]++;
            correct += Math.max(votes[0], Math.max(votes[1], votes[2]));
        }
        return (double) correct / x.length;
    }

    // ---------- Report ----------
    static String buildReport(double[] reg, KMeansResult km, double[] elbow) {
        StringBuilder sb = new StringBuilder();
        sb.append("=== SYNTHETIC FLOWER DATASET ===\n");
        sb.append("Flowers: ").append(x.length).append("   Features: 2   Groups: 3\n\n");
        sb.append("=== DESCRIPTIVE STATISTICS ===\n");
        sb.append(String.format(Locale.US, "%-20s %8s %8s %8s %8s%n", "Feature", "Mean", "Std", "Min", "Max"));
        for (int c = 0; c < 2; c++) {
            double[] col = column(c);
            double mn = Double.MAX_VALUE, mx = -Double.MAX_VALUE;
            for (double v : col) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
            sb.append(String.format(Locale.US, "%-20s %8.2f %8.2f %8.2f %8.2f%n",
                    FEATURES[c], mean(col), std(col), mn, mx));
        }
        sb.append("\n=== CORRELATION & LINEAR REGRESSION ===\n");
        sb.append(String.format(Locale.US, "Pearson r         : %.3f%n", reg[2]));
        sb.append(String.format(Locale.US, "R-squared         : %.3f%n", reg[2] * reg[2]));
        sb.append(String.format(Locale.US, "Model             : width = %.3f * length + %.3f%n", reg[0], reg[1]));
        sb.append("\n=== K-MEANS (k = 3) ===\n");
        for (int c = 0; c < 3; c++) {
            int count = 0;
            for (int l : km.labels) if (l == c) count++;
            sb.append(String.format(Locale.US, "Cluster %d: %3d flowers, centroid = (%.2f, %.2f)%n",
                    c, count, km.centroids[c][0], km.centroids[c][1]));
        }
        sb.append(String.format(Locale.US, "%nInertia           : %.2f%n", km.inertia));
        sb.append(String.format(Locale.US, "Cluster purity    : %.1f%%%n", purity(km, 3) * 100));
        sb.append("\n=== ELBOW (inertia per k) ===\n");
        for (int k = 1; k <= elbow.length; k++)
            sb.append(String.format(Locale.US, "k = %d  ->  %.2f%n", k, elbow[k - 1]));
        return sb.toString();
    }

    // ---------- Scatter chart ----------
    static class ScatterPanel extends JPanel {
        String title;
        int[] colors;
        String[] legend;
        double[][] centroids;
        double[] line; // {slope, intercept} or null

        ScatterPanel(String title, int[] colors, String[] legend, double[][] centroids, double[] line) {
            this.title = title;
            this.colors = colors;
            this.legend = legend;
            this.centroids = centroids;
            this.line = line;
            setBackground(Color.WHITE);
        }

        @Override
        protected void paintComponent(Graphics g0) {
            super.paintComponent(g0);
            Graphics2D g = (Graphics2D) g0;
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            int w = getWidth(), h = getHeight();
            int L = 70, R = 30, T = 55, B = 60;
            double minX = Double.MAX_VALUE, maxX = -Double.MAX_VALUE;
            double minY = Double.MAX_VALUE, maxY = -Double.MAX_VALUE;
            for (double[] p : x) {
                minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
                minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
            }
            double padX = (maxX - minX) * 0.06, padY = (maxY - minY) * 0.06;
            minX -= padX; maxX += padX; minY -= padY; maxY += padY;
            final double fminX = minX, fmaxX = maxX, fminY = minY, fmaxY = maxY;
            final int pw = w - L - R, ph = h - T - B;

            // title
            g.setColor(DARK_RED);
            g.setFont(new Font("SansSerif", Font.BOLD, 18));
            g.drawString(title, L, 32);

            // grid + ticks
            g.setFont(new Font("SansSerif", Font.PLAIN, 12));
            for (int i = 0; i <= 5; i++) {
                int gx = L + pw * i / 5;
                int gy = T + ph * i / 5;
                g.setColor(LIGHT_GRAY);
                g.drawLine(gx, T, gx, T + ph);
                g.drawLine(L, gy, L + pw, gy);
                g.setColor(DARK_GRAY);
                g.drawString(String.format(Locale.US, "%.1f", minX + (maxX - minX) * i / 5), gx - 10, T + ph + 20);
                g.drawString(String.format(Locale.US, "%.1f", maxY - (maxY - minY) * i / 5), L - 38, gy + 4);
            }
            g.setColor(DARK_GRAY);
            g.drawRect(L, T, pw, ph);
            g.setFont(new Font("SansSerif", Font.BOLD, 13));
            g.drawString(FEATURES[0], L + pw / 2 - 50, h - 15);
            g.rotate(-Math.PI / 2);
            g.drawString(FEATURES[1], -(T + ph / 2 + 50), 20);
            g.rotate(Math.PI / 2);

            java.util.function.DoubleUnaryOperator sx = v -> L + (v - fminX) / (fmaxX - fminX) * pw;
            java.util.function.DoubleUnaryOperator sy = v -> T + ph - (v - fminY) / (fmaxY - fminY) * ph;

            // regression line
            if (line != null) {
                g.setColor(RED);
                g.setStroke(new BasicStroke(2.5f));
                double y1 = line[0] * fminX + line[1];
                double y2 = line[0] * fmaxX + line[1];
                g.drawLine((int) sx.applyAsDouble(fminX), (int) sy.applyAsDouble(y1),
                        (int) sx.applyAsDouble(fmaxX), (int) sy.applyAsDouble(y2));
                g.setStroke(new BasicStroke(1f));
            }

            // points
            for (int i = 0; i < x.length; i++) {
                int px = (int) sx.applyAsDouble(x[i][0]);
                int py = (int) sy.applyAsDouble(x[i][1]);
                g.setColor(PALETTE[colors[i]]);
                g.fillOval(px - 6, py - 6, 12, 12);
                g.setColor(Color.WHITE);
                g.drawOval(px - 6, py - 6, 12, 12);
            }

            // centroids
            if (centroids != null) {
                for (double[] c : centroids) {
                    int px = (int) sx.applyAsDouble(c[0]);
                    int py = (int) sy.applyAsDouble(c[1]);
                    g.setColor(Color.WHITE);
                    g.fillRect(px - 9, py - 9, 18, 18);
                    g.setColor(DARK_RED);
                    g.setStroke(new BasicStroke(3f));
                    g.drawRect(px - 9, py - 9, 18, 18);
                    g.drawLine(px - 5, py - 5, px + 5, py + 5);
                    g.drawLine(px - 5, py + 5, px + 5, py - 5);
                    g.setStroke(new BasicStroke(1f));
                }
            }

            // legend
            g.setFont(new Font("SansSerif", Font.PLAIN, 13));
            int ly = T + 12;
            for (int i = 0; i < legend.length; i++) {
                g.setColor(PALETTE[i]);
                g.fillOval(L + pw - 130, ly - 9 + i * 20, 12, 12);
                g.setColor(DARK_GRAY);
                g.drawString(legend[i], L + pw - 112, ly + 2 + i * 20);
            }
        }
    }

    // ---------- Elbow chart ----------
    static class ElbowPanel extends JPanel {
        double[] values;

        ElbowPanel(double[] values) {
            this.values = values;
            setBackground(Color.WHITE);
        }

        @Override
        protected void paintComponent(Graphics g0) {
            super.paintComponent(g0);
            Graphics2D g = (Graphics2D) g0;
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            int w = getWidth(), h = getHeight();
            int L = 80, R = 40, T = 55, B = 60;
            int pw = w - L - R, ph = h - T - B;
            double max = values[0] * 1.05;

            g.setColor(DARK_RED);
            g.setFont(new Font("SansSerif", Font.BOLD, 18));
            g.drawString("Elbow Method (choose k where the curve bends)", L, 32);

            g.setFont(new Font("SansSerif", Font.PLAIN, 12));
            for (int i = 0; i <= 5; i++) {
                int gy = T + ph * i / 5;
                g.setColor(LIGHT_GRAY);
                g.drawLine(L, gy, L + pw, gy);
                g.setColor(DARK_GRAY);
                g.drawString(String.format(Locale.US, "%.0f", max - max * i / 5), L - 50, gy + 4);
            }
            g.setColor(DARK_GRAY);
            g.drawRect(L, T, pw, ph);

            int n = values.length;
            int[] px = new int[n], py = new int[n];
            for (int i = 0; i < n; i++) {
                px[i] = L + (int) (pw * (i + 0.5) / n);
                py[i] = T + ph - (int) (values[i] / max * ph);
                g.setColor(DARK_GRAY);
                g.drawString("k=" + (i + 1), px[i] - 12, T + ph + 20);
            }
            g.setColor(RED);
            g.setStroke(new BasicStroke(3f));
            for (int i = 0; i < n - 1; i++) g.drawLine(px[i], py[i], px[i + 1], py[i + 1]);
            g.setStroke(new BasicStroke(1f));
            for (int i = 0; i < n; i++) {
                g.setColor(DARK_GRAY);
                g.fillOval(px[i] - 7, py[i] - 7, 14, 14);
                g.setColor(Color.WHITE);
                g.drawOval(px[i] - 7, py[i] - 7, 14, 14);
            }
            g.setColor(DARK_GRAY);
            g.setFont(new Font("SansSerif", Font.BOLD, 13));
            g.drawString("Number of clusters", L + pw / 2 - 55, h - 15);
            g.rotate(-Math.PI / 2);
            g.drawString("Inertia", -(T + ph / 2 + 20), 20);
            g.rotate(Math.PI / 2);
        }
    }

    // ---------- Interactive cluster prediction ----------
    static JPanel createPredictionPanel(KMeansResult km) {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBackground(LIGHT_GRAY);
        JPanel card = new JPanel();
        card.setLayout(new BoxLayout(card, BoxLayout.Y_AXIS));
        card.setBackground(Color.WHITE);
        card.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(GRAY),
                BorderFactory.createEmptyBorder(25, 35, 25, 35)));

        JLabel heading = new JLabel("Explore a new flower");
        heading.setForeground(DARK_RED);
        heading.setFont(new Font("SansSerif", Font.BOLD, 22));
        heading.setAlignmentX(Component.LEFT_ALIGNMENT);
        card.add(heading);
        card.add(Box.createVerticalStrut(12));
        JLabel info = new JLabel("Enter petal measurements to find the nearest K-Means cluster.");
        info.setAlignmentX(Component.LEFT_ALIGNMENT);
        card.add(info);
        card.add(Box.createVerticalStrut(20));

        JTextField lengthField = new JTextField("4.5", 12);
        JTextField widthField = new JTextField("1.5", 12);
        JPanel fields = new JPanel(new GridLayout(2, 2, 12, 12));
        fields.setBackground(Color.WHITE);
        fields.add(new JLabel("Petal length (cm):"));
        fields.add(lengthField);
        fields.add(new JLabel("Petal width (cm):"));
        fields.add(widthField);
        fields.setAlignmentX(Component.LEFT_ALIGNMENT);
        card.add(fields);
        card.add(Box.createVerticalStrut(18));

        JLabel result = new JLabel("Your result will appear here.");
        result.setFont(new Font("SansSerif", Font.BOLD, 15));
        result.setForeground(DARK_GRAY);
        result.setAlignmentX(Component.LEFT_ALIGNMENT);
        JButton button = new JButton("Find nearest cluster");
        button.setBackground(RED);
        button.setForeground(Color.WHITE);
        button.setFocusPainted(false);
        button.setAlignmentX(Component.LEFT_ALIGNMENT);
        button.addActionListener(e -> {
            try {
                double length = Double.parseDouble(lengthField.getText().trim().replace(',', '.'));
                double width = Double.parseDouble(widthField.getText().trim().replace(',', '.'));
                if (!Double.isFinite(length) || !Double.isFinite(width) || length <= 0 || width <= 0)
                    throw new NumberFormatException();
                double[] point = {length, width};
                int best = 0;
                for (int i = 1; i < km.centroids.length; i++) {
                    if (dist2(point, km.centroids[i]) < dist2(point, km.centroids[best])) best = i;
                }
                result.setText("Nearest cluster: " + best);
            } catch (NumberFormatException ex) {
                result.setText("Enter two positive numeric measurements.");
            }
        });
        card.add(button);
        card.add(Box.createVerticalStrut(16));
        card.add(result);
        card.add(Box.createVerticalStrut(12));
        JLabel note = new JLabel("Clusters are not guaranteed to match flower species.");
        note.setForeground(GRAY);
        note.setAlignmentX(Component.LEFT_ALIGNMENT);
        card.add(note);
        panel.add(card);
        return panel;
    }

    // ---------- GUI ----------
    static void createGui(double[] reg, KMeansResult km, double[] elbow, String report) {
        JFrame frame = new JFrame("Flower Data Science Project");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        JLabel header = new JLabel("  IRIS LAB  |  Statistics, Regression & K-Means");
        header.setOpaque(true);
        header.setBackground(RED);
        header.setForeground(Color.WHITE);
        header.setFont(new Font("SansSerif", Font.BOLD, 22));
        header.setBorder(BorderFactory.createEmptyBorder(14, 10, 14, 10));

        JTabbedPane tabs = new JTabbedPane();
        tabs.setFont(new Font("SansSerif", Font.BOLD, 14));
        tabs.setBackground(LIGHT_GRAY);
        tabs.setForeground(DARK_GRAY);

        tabs.addTab("Data + Regression",
                new ScatterPanel("True Groups with Regression Line", truth, SPECIES, null,
                        new double[]{reg[0], reg[1]}));
        tabs.addTab("K-Means Clusters",
                new ScatterPanel("K-Means Clusters (k = 3)", km.labels,
                        new String[]{"Cluster 0", "Cluster 1", "Cluster 2"}, km.centroids, null));
        tabs.addTab("Elbow", new ElbowPanel(elbow));
        tabs.addTab("Try a Flower", createPredictionPanel(km));

        JTextArea area = new JTextArea(report);
        area.setEditable(false);
        area.setFont(new Font("Monospaced", Font.PLAIN, 14));
        area.setBackground(LIGHT_GRAY);
        area.setForeground(DARK_GRAY);
        area.setBorder(BorderFactory.createEmptyBorder(15, 15, 15, 15));
        tabs.addTab("Report", new JScrollPane(area));

        frame.setLayout(new BorderLayout());
        frame.add(header, BorderLayout.NORTH);
        frame.add(tabs, BorderLayout.CENTER);
        frame.setSize(950, 700);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        generateData();
        double[] reg = regression(column(0), column(1));
        KMeansResult km = kmeans(3);
        double[] elbow = new double[8];
        for (int k = 1; k <= 8; k++) elbow[k - 1] = kmeans(k).inertia;

        String report = buildReport(reg, km, elbow);
        System.out.println(report);

        SwingUtilities.invokeLater(() -> createGui(reg, km, elbow, report));
    }
}