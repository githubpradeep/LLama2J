import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.reflect.Field;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.*;


// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class Main {

public static void main(String[] args) throws FileNotFoundException {
    List<String> arguments = new ArrayList<>();
    for (String arg : args) {
        arguments.add(arg);
    }

    if (arguments.size() < 1) {
        System.out.println("Usage: <checkpoint_file> [temperature] [steps] [prompt]");
        return;
    }

    String checkpoint = arguments.get(0);
    float temperature = arguments.size() >= 2 ? Float.parseFloat(arguments.get(1)) : 0.9f;
    int steps = arguments.size() >= 3 ? Integer.parseInt(arguments.get(2)) : 256;
    String prompt = arguments.size() >= 4 ? arguments.get(3) : "";



    RandomAccessFile rf = new RandomAccessFile(checkpoint, "r");

    try (FileChannel fileChannel = FileChannel.open(new File(checkpoint).toPath(), StandardOpenOption.READ)) {
        MappedByteBuffer mmap = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
        mmap.order(ByteOrder.LITTLE_ENDIAN);
        float[] data1 = new float[mmap.remaining() / 4];
        mmap.asFloatBuffer().get(data1);
        Config config = Config.load(rf);
        List<Float> data = new ArrayList<>();
        for(int i=0;i<data1.length;i++){
            if (i<7) {
                continue;
            }
            data.add(data1[i]);
        }


        System.out.println("Configuration: " + config);

        TransformerWeights weights = TransformerWeights.init(config, data);

        if (steps <= 0 || steps > config.seq_len) {
            steps = config.seq_len;
        }

        RandomAccessFile tokenFile = new RandomAccessFile("tokenizer.bin", "r");
        Tokenizer tokenizer = Tokenizer.load(tokenFile, config);
        RunState state = new RunState(config);

        List<Integer> promptTokens = !prompt.isEmpty() ? tokenizer.bpeEncode(prompt) : new ArrayList<>();


        long start = 0;
        int next;
        int token = 1;
        int pos = 0;
        System.out.println("<s>");
        while (pos < steps) {
            LLama2.transformer(token, pos, config, state, weights);
            if (pos < promptTokens.size()) {
                next = promptTokens.get(pos);
            } else {
                if (temperature == 0.0) {
                    next = LLama2.argMax(state.logits);
                } else {
                    for (int q = 0; q < config.vocab_size; q++) {
                        state.logits.set(q, state.logits.get(q)/temperature);
                    }
                    LLama2.softmax(state.logits.subList(0, config.vocab_size));
                    next = LLama2.sample(state.logits, config.vocab_size);
                }
            }

            String tokenStr = token == 1 && tokenizer.vocab.get(next).startsWith(" ")
                    ? tokenizer.vocab.get(next).substring(1)
                    : tokenizer.vocab.get(next);
            System.out.print(tokenStr);
            System.out.flush();

            token = next;
            pos++;
            if (start == 0) {
                start = System.currentTimeMillis();
            }
        }

        long end = System.currentTimeMillis();
        System.out.println("\nachieved tok/s: " + ((steps - 1) / (end - start)) * 1000.0);
    } catch (IOException e) {
        e.printStackTrace();
    }
}

}

class LLama2 {

    public static int sample(List<Float> probabilities, int n) {
        // Sample index from probabilities, they must sum to 1
        Random random = new Random();
        float r = random.nextFloat();
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probabilities.get(i);
            if (r < cdf) {
                return i;
            }
        }
        return n - 1; // In case of rounding errors
    }

    public  static int readInt(RandomAccessFile f) {
        try {
            byte[] bytes = new byte[4];
            f.read(bytes, 0, 4);
            return ((bytes[0] & 0xFF) |
                    ((bytes[1] & 0xFF) << 8) |
                    ((bytes[2] & 0xFF) << 16) |
                    ((bytes[3] & 0xFF) << 24));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static float readFloat(RandomAccessFile f) {
        byte[] buffer = new byte[4];
        try {
            f.read(buffer, 0, 4);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        int intBits = ((buffer[3] & 0xFF) << 24) | ((buffer[2] & 0xFF) << 16) | ((buffer[1] & 0xFF) << 8) | (buffer[0] & 0xFF);
        return Float.intBitsToFloat(intBits);

    }

    public static String readString(RandomAccessFile f, int len) {
        try {
            byte[] bytes = new byte[len];
            f.read(bytes, 0, len);
            String s =  new String(bytes, StandardCharsets.UTF_8);
            return s;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void accum(List<Float> a, List<Float> b) {
        int length = Math.min(a.size(), b.size());
        for (int i = 0; i < length; i++) {
            a.set(i, a.get(i)+ b.get(i));
        }
    }

    public static void rmsnorm(List<Float> o, List<Float> xo, List<Float> weight, int size) {
        assert size == o.size();

        // Calculate sum of squares
        float ss = 0.0f;
        for (int i = 0; i < size; i++) {
            float x = xo != null ? xo.get(i) : o.get(i);
            ss += x * x;
        }


        ss /= o.size();
        ss += 1e-5f;
        ss = 1.0f / (float) Math.sqrt(ss);

        // Normalize and scale
        for (int j = 0; j < size; j++) {
            float x = xo != null ? xo.get(j) : o.get(j);
            o.set(j, weight.get(j) * ss * x);
        }
    }

    public static void softmax(List<Float> x) {
        // Find max value (for numerical stability)
        float maxVal = x.get(0);
        for (float x_i : x) {
            maxVal = Math.max(maxVal, x_i);
        }

        // Exp and sum
        float sum = 0.0f;
        for (int i = 0; i < x.size(); i++) {
            float v = (float) Math.exp(x.get(i) - maxVal);
            x.set(i, v);
            sum += v;
        }

        // Normalize
        for (int i = 0; i < x.size(); i++) {
            x.set(i,  x.get(i)/sum);
        }


    }

    public static int strLookup(String str, List<String> vocab) {
        return vocab.indexOf(str);
    }

    public static int argMax(List<Float> v) {
        int index = -1;
        float maxVal = Float.NEGATIVE_INFINITY;

        for (int i = 0; i < v.size(); i++) {
            if (v.get(i) > maxVal) {
                maxVal = v.get(i);
                index = i;
            }
        }

        return index;
    }

    public static void matmul(List<Float> xout, List<Float> x, List<Float> w, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        // By far the most amount of time is spent inside this little function
        assert d == xout.size();
        assert n == x.size();

        for (int i = 0; i < d; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += w.get(i * n + j) * x.get(j);
            }
            xout.set(i, sum);
        }
    }

    public static float dot(List<Float> q, List<Float> k) {
        assert q.size() == k.size();
        float result = 0.0f;
        for (int i = 0; i < q.size(); i++) {
            result += q.get(i) * k.get(i);
        }
        return result;
    }



    public static  List<Float> copyOfRange(List<Float> list, int from, int to) {
        if (from < 0 || from > list.size() || to < 0 || to > list.size() || from > to)
            throw new IllegalArgumentException("Illegal extraction bounds");
        return list.subList(from, to);
    }

    public static void transformer(int token, int pos, Config p, RunState s, TransformerWeights w) {
        // A few convenience variables
        List<Float> x = s.x;
        int dim = p.dim;
        int hiddenDim = p.hidden_dim;
        int headSize = dim / p.n_heads;

        // Copy the token embedding into x
        int tokenStart = token * dim;
        for (int i = 0; i < dim; i++) {
            x.set(i, w.token_embedding_table.get(tokenStart + i));
        }


        // Pluck out the "pos" row of freqCisReal and freqCisImag
        List<Float> freqCisRealRow = LLama2.copyOfRange(w.freq_cis_real, pos * headSize / 2, (pos + 1) * headSize / 2);
        List<Float>  freqCisImagRow = LLama2.copyOfRange(w.freq_cis_imag, pos * headSize / 2, (pos + 1) * headSize / 2);


        // Forward all the layers
        for (int l = 0; l < p.n_layers; l++) {
            // Attention RMSNorm

            rmsnorm(s.xb, x, LLama2.copyOfRange(w.rms_att_weight, l * dim, (l + 1) * dim), dim);




            // QKV matmuls for this position
            matmul(s.q, s.xb, LLama2.copyOfRange(w.wq, l * dim * dim, (l + 1) * dim * dim), dim, dim);
            matmul(s.k, s.xb, LLama2.copyOfRange(w.wk, l * dim * dim, (l + 1) * dim * dim), dim, dim);
            matmul(s.v, s.xb, LLama2.copyOfRange(w.wv, l * dim * dim, (l + 1) * dim * dim), dim, dim);

            // Apply RoPE rotation to the q and k vectors for each head
            for (int h = 0; h < p.n_heads; h++) {
                int qStart = h * headSize;
                int kStart = h * headSize;
                for (int i = 0; i < headSize; i += 2) {
                    float q0 = s.q.get(qStart + i);
                    float q1 = s.q.get(qStart + i + 1);
                    float k0 = s.k.get(kStart + i);
                    float k1 = s.k.get(kStart + i + 1);
                    float fcr = freqCisRealRow.get(i / 2);
                    float fci = freqCisImagRow.get(i / 2);
                    s.q.set(qStart + i, q0 * fcr - q1 * fci);
                    s.q.set(qStart + i + 1, q0 * fci + q1 * fcr);
                    s.k.set(kStart + i, k0 * fcr - k1 * fci);
                    s.k.set(kStart + i + 1, k0 * fci + k1 * fcr);
                }
            }

            // Save key, value at this time step (pos) to our kv cache
            List<Float> keyCacheRow = s.key_cache.get(l).get(pos);
            List<Float> valueCacheRow = s.value_cache.get(l).get(pos);

            for (int i=0; i< s.k.size();i++) {
                keyCacheRow.set(i, s.k.get(i));
            }
            for (int i=0; i< s.v.size();i++) {
                valueCacheRow.set(i, s.v.get(i));
            }

            // Multi-head attention. Iterate over all heads
            for (int h = 0; h < p.n_heads; h++) {
                // Get the query vector for this head
                int qStart = h * headSize;
                List<Float> q = s.q.subList(qStart, qStart + headSize);
                List<Float> xb = s.xb.subList(qStart, qStart + headSize);

                // Attention scores for this head
                List<Float> att = s.att.get(h);

                // Iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++) {
                    // Get the key vector for this head and at this timestep
                    int kStart = h * headSize;
                    List<List<Float>> lthLayer = s.key_cache.get(l);
                    List<Float> thTimestep = lthLayer.get(t);
                    List<Float> k = thTimestep.subList(kStart, kStart+headSize);


                    // Calculate the attention score as the dot product of q and k
                    float score = dot(q, k) / (float) Math.sqrt(headSize);

                    // Save the score to the attention buffer
                    att.set(t, score);
                }

                // Softmax the scores to get attention weights, from 0..pos inclusively
                softmax(att.subList(0, pos+1));

                // Weighted sum of the values, store back into xb

                for(int i=0; i<headSize;i++) {
                    xb.set(i, (float) 0.0f);
                }



                for (int t = 0; t <= pos; t++) {
                    // Get the value vector for this head and at this timestep
                    int vStart = h * headSize;


                    List<List<Float>> lthLayer = s.value_cache.get(l);
                    List<Float> thTimestep = lthLayer.get(t);
                    List<Float> v = thTimestep.subList(vStart, vStart+headSize);





                    // Get the attention weight for this timestep

                    float a = att.get(t);


                    // Accumulate the weighted value into xb
                    for (int i = 0; i < headSize; i++) {
                        xb.set( i, xb.get(i) + a * v.get(i));
                    }
                }
            }



            // Final matmul to get the output of the attention
            matmul(s.xb2, s.xb, LLama2.copyOfRange(w.wo, l * dim * dim, (l + 1) * dim * dim), dim, dim);




            // Residual connection back into x
            accum(x, s.xb2);



            //System.out.println(x);

            // FFN RMSNorm
            rmsnorm(s.xb, x, LLama2.copyOfRange(w.rms_ffn_weight, l * dim, (l + 1) * dim), dim);

            // Now for FFN in PyTorch, we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // First calculate self.w1(x) and self.w3(x)
            matmul(s.hb, s.xb, LLama2.copyOfRange(w.w1, l * dim * hiddenDim, (l + 1) * dim * hiddenDim), dim, hiddenDim);
            matmul(s.hb2, s.xb, LLama2.copyOfRange(w.w3, l * dim * hiddenDim, (l + 1) * dim * hiddenDim), dim, hiddenDim);

            // F.silu; silu(x) = x * σ(x), where σ(x) is the logistic sigmoid
            for (int i = 0; i < hiddenDim; i++) {
                s.hb.set(i, s.hb.get(i) * 1.0f / (1.0f + (float) Math.exp(-s.hb.get(i))));
            }

            // Elementwise multiply with w3(x)
            for (int i = 0; i < hiddenDim; i++) {
                s.hb.set(i, s.hb.get(i) * s.hb2.get(i));
            }

            // Final matmul to get the output of the FFN
            matmul(s.xb, s.hb, LLama2.copyOfRange(w.w2, l * dim * hiddenDim, (l + 1) * dim * hiddenDim), hiddenDim, dim);

            // Residual connection
            accum(x, s.xb);
        }

        // Final RMSNorm
        rmsnorm(x, null, w.rms_final_weight, dim);

        // Classifier into logits

        matmul(s.logits, x, w.wcls, dim, p.vocab_size);

    }

    public static long timeInMs() {
        // Return time in milliseconds, for benchmarking the model speed
        return Instant.now().toEpochMilli();
    }


}

class Ptr {
    List<Float> x;
    int total;

    public Ptr(List<Float> x, int total) {
        this.x = x;
        this.total = total;
    }


    public List<Float> align(int size) {
        this.total += size;
        List<Float> ret = this.x.subList(0, size);
        this.x = this.x.subList(size, this.x.size());
        return ret;
    }
}

class  Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    boolean shared_weight;

    public static Config load(RandomAccessFile randomAccessFile) throws IOException {
        try  {
            Config conf = new Config();
            conf.dim = LLama2.readInt(randomAccessFile);
            conf.hidden_dim = LLama2.readInt(randomAccessFile);
            conf.n_layers = LLama2.readInt(randomAccessFile);
            conf.n_heads = LLama2.readInt(randomAccessFile);
            conf.n_kv_heads = LLama2.readInt(randomAccessFile);

            int vocab_size = LLama2.readInt(randomAccessFile);
            conf.vocab_size = Math.abs(vocab_size);
            conf.shared_weight = vocab_size > 0;
            conf.seq_len = LLama2.readInt(randomAccessFile);

            return conf;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    @Override
    public String toString()
    {
        StringBuilder result = new StringBuilder();
        String newLine = System.getProperty("line.separator");

        result.append( this.getClass().getName() );
        result.append( " Object {" );
        result.append(newLine);

        //determine fields declared in this class only (no fields of superclass)
        Field[] fields = this.getClass().getDeclaredFields();

        //print field names paired with their values
        for ( Field field : fields  ) {
            result.append("  ");
            try {
                result.append( field.getName() );
                result.append(": ");
                //requires access to private field:
                result.append( field.get(this) );
            } catch ( IllegalAccessException ex ) {
                System.out.println(ex);
            }
            result.append(newLine);
        }
        result.append("}");

        return result.toString();
    }
}

class RunState {
    // current wave of activations
    public List<Float> x;        // activation at the current time stamp (dim,)
    public List<Float> xb;       // same, but inside a residual branch (dim,)
    public List<Float> xb2;      // an additional buffer just for convenience (dim,)
    public List<Float> hb;       // buffer for the hidden dimension in the ffn (hidden_dim,)
    public List<Float> hb2;      // buffer for the hidden dimension in the ffn (hidden_dim,)
    public List<Float> q;        // query (dim,)
    public List<Float> k;        // key (dim,)
    public List<Float> v;        // value (dim,)
    public List<List<Float>> att; // buffer for scores/attention values (n_heads, seq_len)
    public List<Float> logits;   // output logits

    // kv cache
    public List<List<List<Float>>> key_cache;   // (layer, seq_len, dim)
    public List<List<List<Float>>> value_cache; // (layer, seq_len, dim)

    // Constructor
    public RunState(Config p) {
        x = new ArrayList<>();
        xb = new ArrayList<>();
        xb2 = new ArrayList<>();
        hb = new ArrayList<>();
        hb2 = new ArrayList<>();
        q = new ArrayList<>();
        k = new ArrayList<>();
        v = new ArrayList<>();
        att = new ArrayList<>();
        logits = new ArrayList<>();
        key_cache = new ArrayList<>();
        value_cache = new ArrayList<>();

        for (int i = 0; i < p.dim; i++) {
            x.add(0.0f);
            xb.add(0.0f);
            xb2.add(0.0f);
            q.add(0.0f);
            k.add(0.0f);
            v.add(0.0f);
        }

        for (int i = 0; i < p.hidden_dim; i++) {
            hb.add(0.0f);
            hb2.add(0.0f);
        }

        for (int i = 0; i < p.n_heads; i++) {
            List<Float> attRow = new ArrayList<>();
            for (int j = 0; j < p.seq_len; j++) {
                attRow.add(0.0f);
            }
            att.add(attRow);
        }

        for (int i = 0; i < p.vocab_size; i++) {
            logits.add(0.0f);
        }

        for (int i = 0; i < p.n_layers; i++) {
            List<List<Float>> keyCacheLayer = new ArrayList<>();
            List<List<Float>> valueCacheLayer = new ArrayList<>();

            for (int j = 0; j < p.seq_len; j++) {
                List<Float> keyCacheRow = new ArrayList<>();
                List<Float> valueCacheRow = new ArrayList<>();
                for (int k = 0; k < p.dim; k++) {
                    keyCacheRow.add(0.0f);
                    valueCacheRow.add(0.0f);
                }
                keyCacheLayer.add(keyCacheRow);
                valueCacheLayer.add(valueCacheRow);
            }

            key_cache.add(keyCacheLayer);
            value_cache.add(valueCacheLayer);
        }
    }
}


class TransformerWeights {
    // token embedding table
    public List<Float> token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    public List<Float> rms_att_weight; // (layer, dim) rmsnorm weights
    public List<Float> rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    public List<Float> wq; // (layer, dim, dim)
    public List<Float> wk; // (layer, dim, dim)
    public List<Float> wv; // (layer, dim, dim)
    public List<Float> wo; // (layer, dim, dim)
    // weights for ffn
    public List<Float> w1; // (layer, hidden_dim, dim)
    public List<Float> w2; // (layer, dim, hidden_dim)
    public List<Float> w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    public List<Float> rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    public List<Float> freq_cis_real; // (seq_len, dim/2)
    public List<Float> freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    public List<Float> wcls;

    public TransformerWeights() {
        token_embedding_table = new ArrayList<>();
        rms_att_weight = new ArrayList<>();
        rms_ffn_weight = new ArrayList<>();
        wq = new ArrayList<>();
        wk = new ArrayList<>();
        wv = new ArrayList<>();
        wo = new ArrayList<>();
        w1 = new ArrayList<>();
        w2 = new ArrayList<>();
        w3 = new ArrayList<>();
        rms_final_weight = new ArrayList<>();
        freq_cis_real = new ArrayList<>();
        freq_cis_imag = new ArrayList<>();
        wcls = new ArrayList<>();
    }

    // Constructor
    public static TransformerWeights init(Config p, List<Float> f) {
        TransformerWeights ret = new TransformerWeights();
        int head_size = p.dim / p.n_heads;

        Ptr ptr = new Ptr(f, 0);
        ret.token_embedding_table = ptr.align(p.vocab_size * p.dim);
        ret.rms_att_weight = ptr.align(p.n_layers * p.dim);
        ret.wq = ptr.align(p.n_layers * p.dim * p.dim);
        ret.wk = ptr.align(p.n_layers * p.dim * p.dim);
        ret.wv = ptr.align(p.n_layers * p.dim * p.dim);
        ret.wo = ptr.align(p.n_layers * p.dim * p.dim);
        ret.rms_ffn_weight = ptr.align(p.n_layers * p.dim);
        ret.w1 = ptr.align(p.n_layers * p.hidden_dim * p.dim);
        ret.w2 = ptr.align(p.n_layers * p.dim * p.hidden_dim);
        ret.w3 = ptr.align(p.n_layers * p.hidden_dim * p.dim);
        ret.rms_final_weight = ptr.align(p.dim);
        ret.freq_cis_real = ptr.align(p.seq_len * head_size / 2);
        ret.freq_cis_imag = ptr.align(p.seq_len * head_size / 2);

        if (!p.shared_weight) {
            ret.wcls = ptr.align(p.dim * p.vocab_size);
        } else {
            ret.wcls = ret.token_embedding_table;
        }

        assert ptr.total == f.size();
        return ret;
    }
}


class Tokenizer {
    public List<String> vocab;
    public List<Float> vocab_scores;
    public int max_token_length;

    public Tokenizer() {
        this.vocab = new ArrayList<>();
        this.vocab_scores = new ArrayList<>();
        this.max_token_length =0;
    }

    public static Tokenizer load(RandomAccessFile file, Config config) throws IOException {
        Tokenizer tokenizer = new Tokenizer();
        try  {
            tokenizer.max_token_length = LLama2.readInt(file);
            for (int i = 0; i < config.vocab_size; i++) {
                tokenizer.vocab_scores.add(LLama2.readFloat(file));
                int len =  LLama2.readInt(file);
                tokenizer.vocab.add(LLama2.readString(file, len));
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return tokenizer;
    }





    public List<Integer> bpeEncode(String text) {
        List<Integer> tokens = new ArrayList<>();
        StringBuilder strBuffer = new StringBuilder();

        // First encode every individual byte in the input string
        for (int i = 0; i < text.length(); i++) {
            strBuffer.setLength(0); // Clear the StringBuilder
            strBuffer.append(text.charAt(i));
            int id = LLama2.strLookup(strBuffer.toString(), vocab);
            if (id != -1) {
                tokens.add(id);
            } else {
                throw new IllegalArgumentException("Token not found in vocab.");
            }
        }

        // Merge the best consecutive pair each iteration, according to the scores in vocab_scores
        while (true) {
            double bestScore = -1e10;
            int bestId = 0;
            Integer bestIdx = null;

            for (int i = 0; i < (tokens.size() - 1); i++) {
                // Check if we can merge the pair (tokens[i], tokens[i+1])
                strBuffer.setLength(0); // Clear the StringBuilder
                strBuffer.append(vocab.get(tokens.get(i)));
                strBuffer.append(vocab.get(tokens.get(i + 1)));
                int id = LLama2.strLookup(strBuffer.toString(), vocab);
                if (id != -1) {
                    if (vocab_scores.get(id) > bestScore) {
                        // This merge pair exists in vocab! Record its score and position
                        bestScore = vocab_scores.get(id);
                        bestId = id;
                        bestIdx = i;
                    }
                }
            }

            if (bestIdx == null) {
                return tokens; // We couldn't find any more pairs to merge, so we're done
            } else {
                // Merge the consecutive pair (bestIdx, bestIdx+1) into new token bestId
                tokens.set(bestIdx, bestId);
                // Delete token at position bestIdx+1, shift the entire sequence back 1
                for (int i = (bestIdx + 1); i < (tokens.size() - 1); i++) {
                    tokens.set(i, tokens.get(i + 1));
                }
                tokens.remove(tokens.size() - 1); // Token length decreased
            }
        }
    }

}
