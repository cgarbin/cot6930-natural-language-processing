import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;

/**
 * OpenNLP parts of speech (POS) example.
 */
public class PartsOfSpeechExample {

  public static void testPartsOfSpeech(String document) {
    POSTaggerME tagger = new POSTaggerME(getPOSModel());

    // Use ME tokenization for POS because ME tokenization does a better job of
    // returning tokens that are meaningful for humans.
    List<TokenizationExample.TokenizedSentence> tokens = TokenizationExample.getMETokens(document);
    for (TokenizationExample.TokenizedSentence ts : tokens) {
      String[] tags = tagger.tag(ts.tokens);
      printTags(ts.sentence, ts.tokens, tags);
    }
  }

  /**
   * Return the OpenNLP English maximum entropy POS model, or null if can't load
   * it.
   */
  private static POSModel getPOSModel() {
    try (InputStream modelIn = new FileInputStream("opennlp-models/en-pos-maxent.bin")) {
      return new POSModel(modelIn);
    } catch (IOException e) {
      e.printStackTrace();
    }
    return null;
  }

  private static void printTags(String sentence, String[] tokens, String[] tags) {
    System.out.println("\n" + sentence + "\n");
    for (int i = 0; i < tokens.length; i++) {
      System.out.println(String.format("[%02d] %-15s %s", i, tokens[i], tags[i]));
    }

  }

}
