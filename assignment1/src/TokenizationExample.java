import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.tokenize.WhitespaceTokenizer;

/**
 * OpenNLP tokenization example.
 */
public class TokenizationExample {

  /**
   * Class to hold a sentence and its tokens.
   */
  public static class TokenizedSentence {
    public String sentence;
    public String[] tokens;

    TokenizedSentence(String s, String[] t) {
      sentence = s;
      tokens = t;
    }
  }

  /**
   * Tokenize the document using the simple tokenizer, based on character classes
   * ("sequences of the same character class are tokens").
   */
  public static List<TokenizedSentence> getSimpleTokens(String document) {
    return getTokens(document, SimpleTokenizer.INSTANCE);
  }

  /**
   * Tokenize the document using the whitespace tokenizer ("non whitespace
   * sequences are identified as tokens").
   */
  public static List<TokenizedSentence> getWhitespaceTokens(String document) {
    return getTokens(document, WhitespaceTokenizer.INSTANCE);
  }

  /**
   * Tokenize the document using the maximum entropy (ME) tokenizer ("detects
   * token boundaries based on probability model").
   */
  public static List<TokenizedSentence> getMETokens(String document) {
    return getTokens(document, new TokenizerME(getTokenizerModel()));
  }

  public static void testSimpleTokens(String document) {
    printTokens(getSimpleTokens(document));
  }

  public static void testWhitespaceTokens(String document) {
    printTokens(getWhitespaceTokens(document));
  }

  public static void testMETokens(String document) {
    printTokens(getMETokens(document));
  }

  /**
   * Tokenize all sentences in the document with the given tokenizer, return the
   * token and the sentence from which they were extracted.
   */
  private static List<TokenizedSentence> getTokens(String document, Tokenizer tokenizer) {
    String[] sentences = SentenceDetectionExample.getSentences(document);
    ArrayList<TokenizedSentence> tokens = new ArrayList<>();
    for (String s : sentences) {
      String[] t = tokenizer.tokenize(s);
      tokens.add(new TokenizedSentence(s, t));
    }
    return tokens;
  }

  /**
   * Print a sentence and its tokens.
   */
  private static void printTokens(List<TokenizedSentence> tokens) {
    for (TokenizedSentence ts : tokens) {
      System.out.println("\n" + ts.sentence);
      Utils.printStringArray(ts.tokens);
    }
  }

  /**
   * Return the OpenNLP English tokenizer, or null if can't load it.
   */
  private static TokenizerModel getTokenizerModel() {
    try (InputStream modelIn = new FileInputStream("opennlp-models/en-token.bin")) {
      return new TokenizerModel(modelIn);
    } catch (IOException e) {
      e.printStackTrace();
    }
    return null;
  }
}
