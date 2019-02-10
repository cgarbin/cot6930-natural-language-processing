public class Main {

  public static void main(String[] args) {
    // The document we will use for all tests.
    String document = Utils.getSampleText();

    System.out.println("-----\nSentence detection\n\n");
    SentenceDetectionExample.testSentences(document);

    System.out.println("\n\n-----\nTokenization - simple\n");
    TokenizationExample.testSimpleTokens(document);
    System.out.println("\n\n-----\nTokenization - whitespace\n");
    TokenizationExample.testWhitespaceTokens(document);
    System.out.println("\n\n-----\nTokenization - ME (maximum entropy - probabilistic)\n");
    TokenizationExample.testMETokens(document);

    System.out.println("\n\n-----\nParts of speech (POS)\n\n");
    PartsOfSpeechExample.testPartsOfSpeech(document);

    System.out.println("\n\n-----\nNamed entity detection\n\n");
    NamedEntityExample.testNameFinder(document);
  }
}
