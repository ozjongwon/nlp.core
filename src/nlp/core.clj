;;;
;;; * NLP SUMMARY
;;;
;;; Application Areas of NLP
;;; - Searching
;;; - Machine Translation
;;; - Summation
;;; - Named Entity Recognition - NER
;;; - Information Grouping
;;; - Part Of Speech tagging (POS)
;;; - Sentiment Analysis
;;; - Answering Queries
;;; - Speech recognition
;;; - Natural Language Generation
;;;
;;; Morpheme - the minimal unit of text which has meaning. Ex) prefixes and suffixes
;;; POS label - assign labels to words and morphemes.
;;; Stemming - finding the word stem.
;;; Lemmatization - determine the lemma, the base form
;;; Corereferences resolution - determine the relationship between words in sentences.
;;; Word Sense Disambiguation - find the intended meaning.
;;; Morphology - study of the structure of words
;;; Parts of Text - words, sentences, paragraphs, etc OR sometimes tokens == words
;;; Claassification - Assign labels to information found in text or documents:
;;;  - labels are known - classification.
;;;  - labels are unknown - clustering.
;;; Categorization - assign a text element to categories.
;;;
;;;-------------------------------------------------------------------------------------------
;;;
;;; * PTBTokenizer - CoreLabelTokenFactory Vs. WordTokenFactory
;;; * Annotators - extract relationship using Annotators with their operations
;;;  (aka text analytics). Ex) annotators == :tokenize :ssplit :parse ==> tokenize,
;;;  splits to sentences, and syntactic analysis by parsing
;;;
;;; POS - https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
;;; - CC - Coordinating conjunction
;;; - CD - Cardinal number
;;; - DT - Determiner
;;; - DT - Determiner
;;; - EX - Existential there
;;; - FW - Foreign word
;;; - IN - Preposition or subordinating conjunction
;;; - JJ - Adjective
;;; - JJR -Adjective, comparative
;;; - JJS - Adjective, superlative
;;; - LS - List item marker
;;; - MD - Modal
;;; - NN - Noun, singular, or mass
;;; - NNP - Proper noun, singular
;;; - NNPS - Proper noun, plural
;;; - NNS - Noun, plural
;;; - PDT - Predeterminer
;;; - POS - Possessive ending
;;; - PRP - Personal pronoun
;;; - PRP$ - Possessive pronoun
;;; - RB - Adverb
;;; - RBR - Adverb, comparative
;;; - RBS - Adverb, superlative
;;; - RP - Particle
;;; - SYM - Symbol
;;; - TO - To
;;; - UH - Interjection
;;; - VB - Verb, base form
;;; - VBD - Verb, past tense
;;; - VBG - Verb, gerunds or present participle
;;; - VBN - Verb, past participle
;;; - VBP - Verb, non-third person singular present
;;; - VBZ - Verb, third person singular present
;;; - WDT - Wh-determiner
;;; - WP - Wh-pronoun
;;; - WP$ - Possessive wh-pronoun
;;; - WRB- Wh-adverb
;;;
(ns nlp.core
  (:gen-class)
  (:require
   [clojure.string :refer [join]]
   [clojure.set :refer [intersection]]
   [nlp.utils :refer [find-in-coll make-keyword]]
   [nlp.records :refer [make-token-result TokenBasedResult get-result-prototype
                        token-ann->token-map token-based-result->annotation-class]])
  (:import
   [nlp.records TokenizeResult PosResult LemmaResult NerResult]
   [java.util Properties]
   [clojure.lang Reflector]
   ;;[clojure.reflect Constructor]
   ;;clojure.lang.Reflector
   [java.io StringReader]
   ;; Tokenizers
   ;;[edu.stanford.nlp.international.arabic.process ArabicTokenizer]
   ;;[edu.stanford.nlp.trees.international.pennchinese CHTBTokenizer]
   [edu.stanford.nlp.trees LabeledScoredTreeNode TreeCoreAnnotations$TreeAnnotation]
   [edu.stanford.nlp.semgraph SemanticGraphCoreAnnotations$EnhancedPlusPlusDependenciesAnnotation]
   ;;[edu.stanford.nlp.international.french.process FrenchTokenizer]
   [edu.stanford.nlp.process WordTokenFactory CoreLabelTokenFactory
    DocumentPreprocessor PTBTokenizer TokenizerFactory
    ;; LexerTokenizer TokenizerAdapter WhitespaceTokenizer WordSegmentingTokenizer
    ]
   ;;[edu.stanford.nlp.ie.machinereading.domains.ace.reader RobustTokenizer]
   ;;[edu.stanford.nlp.international.spanish.process SpanishTokenizer]

   [edu.stanford.nlp.ling CoreAnnotations$SentencesAnnotation CoreAnnotations$TextAnnotation
    CoreAnnotations$NamedEntityTagAnnotation
    CoreAnnotations$PartOfSpeechAnnotation CoreAnnotations$LemmaAnnotation
    CoreAnnotations$MentionsAnnotation CoreAnnotations$NamedEntityTagAnnotation
    CoreLabel TaggedWord Word SentenceUtils]
   [edu.stanford.nlp.coref CorefCoreAnnotations$CorefChainAnnotation]
   ;; [edu.stanford.nlp.dcoref CorefCoreAnnotations$CorefChainAnnotation]
   [edu.stanford.nlp.pipeline Annotation StanfordCoreNLP CoreDocument]

   [edu.stanford.nlp.tagger.maxent MaxentTagger]
   [edu.stanford.nlp.parser.lexparser LexicalizedParser])
  (:gen-class :main true))



;;;
;;; StanfordCoreNLP Pipeline
;;; https://stanfordnlp.github.io/CoreNLP/annotators.html
;;;

(defonce key->property-dependency
  {:tokenize []
   :docdate []
   :cleanxml ["tokenize"]
   :ssplit ["tokenize"]                 ; sentence detection
   :pos ["tokenize" "ssplit"]
   :parse ["tokenize" "ssplit"]
   :lemma ["tokenize" "ssplit" "pos"]
   :regexner ["tokenize" "ssplit" "pos"]
   :depparse ["tokenize" "ssplit" "pos"]
   :ner ["tokenize" "ssplit" "pos" "lemma"] ; Named Entity Recognition
   :entitylink ["tokenize" "ssplit" "pos" "lemma"  "ner"]
   :sentiment ["tokenize" "ssplit" "pos" "parse"]
   :dcoref ["tokenize" "ssplit" "pos" "lemma"  "ner" "parse"]
   :coref ["tokenize" "ssplit" "pos" "lemma"  "ner"]
   :kbp ["tokenize" "ssplit" "pos" "lemma"]
   :quote ["tokenize" "ssplit" "pos" "lemma" "ner" "depparse"]})

(defn- %check-parse-dependeny-when-opt [args opt]
  (when (find-in-coll args opt)
    (let [parse? (find-in-coll args :parse)
          dparse? (find-in-coll args :dparse)]
      (when (and parse? dparse?)
        (throw (Exception. "Only one of parse or depparse is allowed")))
      (when (not (or parse? dparse?))
        (throw (Exception. "One of parse or depparse is required"))))))

(defn- maybe-check-coref-dependeny [args]
  (when (find-in-coll args :coref)
    (%check-parse-dependeny-when-opt args :coref)))

(defn- maybe-check-kbp-dependency [args]
  (when (find-in-coll args :kbp)
    (%check-parse-dependeny-when-opt args :kbp)
    (when-not (find-in-coll args :coref)
      (throw (Exception. "Option :coref is required")))))

(defn make-annotators-opts
  ([args]
   ;; Special case checking
   (maybe-check-coref-dependeny args)
   (maybe-check-kbp-dependency args)
   (make-annotators-opts args []))
  ([[k & more-ks] result]
   (if (nil? k)
     (distinct result)
     (if-let [opts-found (key->property-dependency k)]
       (recur more-ks (into result (conj opts-found (name k))))
       (throw (Exception. (str "Unknown key: " k "!")))))))

(let [existing-core-nlp (atom nil)
      existing-opts-set (atom #{})]
  (defn make-pipeline [& annotators-keys]
    (let [annotators-opts (make-annotators-opts annotators-keys)
          annotators-opts-set (set annotators-opts)]
     (if (and @existing-core-nlp
              (= (intersection @existing-opts-set annotators-opts-set)
                 annotators-opts-set))
       @existing-core-nlp
       (let [new-core-nlp (StanfordCoreNLP. (doto (Properties.)
                                              (.put "annotators" (join \, annotators-opts)))
                                            true)]
         (reset! existing-core-nlp new-core-nlp)
         (reset! existing-opts-set annotators-opts-set)
         new-core-nlp)))))

;;; Operations
(defmulti annotator-key->execute-operation (fn [k _] k))

;; :parse
(defrecord ParseResult [tree pos dependency])

(def atom? (complement coll?))
(defn convert-tree
  ([tr kfn]
   (convert-tree tr kfn identity))
  ([[k v & more-nodes] kfn vfn]
   (if (nil? k)
     nil
     (cons (kfn k)
           (cons (if (atom? v)
                   (vfn v)
                   (convert-tree v kfn vfn))
                 (map #(convert-tree % kfn vfn) more-nodes))))))

(defn- tree->parse-tree [tree-node]
  (convert-tree (read-string (.toString tree-node)) make-keyword name))

(defn- tree->pos [tree-node]
  (->> (.taggedLabeledYield tree-node)
       (mapv #(vector (.word %) (make-keyword (.tag %))))
       (merge (hash-map))))

(defmethod annotator-key->execute-operation :parse [k ann]
  (mapv #(let [tree-node (.get % TreeCoreAnnotations$TreeAnnotation)]
           (map->ParseResult {:tree (tree->parse-tree tree-node)
                              :pos (tree->pos tree-node)
                              :dependency (.get % SemanticGraphCoreAnnotations$EnhancedPlusPlusDependenciesAnnotation)}))
        ;; FIXME:
        ;; https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/semgraph/SemanticGraph.html
        (.get ann CoreAnnotations$SentencesAnnotation)))
;;;;;;

(defn- sentence-ann->token-based-result [sentence-ann result-class]
  (->> (token-based-result->annotation-class (get-result-prototype result-class))
       (.get sentence-ann)
       (mapv #(make-token-result result-class %))))

(defn- annotation->token-based-results [ann result-class]
  (->> (.get ann CoreAnnotations$SentencesAnnotation)
       (mapv #(sentence-ann->token-based-result % result-class))))

(defmethod annotator-key->execute-operation :tokenize [k ann]
  (annotation->token-based-results ann TokenizeResult))

;; :pos
(defmethod annotator-key->execute-operation :pos [k ann]
  (annotation->token-based-results ann PosResult))

;; :lemma
(def lemma-paragraph "Similar to stemming is Lemmatization. This is the process of finding its lemma, its form as found in a dictionary.")

(def pos-paragraph "Bill used the force to force the manger to tear the bill in two.")
(def pos-paragraph2 "AFAIK she H8 cth! BTW had a GR8 tym at the party BBIAM.")
(def pos-paragraph3 "Whether \"Blue\" was correct or not (it's not) is debatable.")

(def pos-paragraph4 "The voyage of the Abraham Lincoln was for a long time marked by no special incident. But one circumstance happened which showed the wonderful dexterity of Ned Land, and proved what confidence we might place in him. The 30th of June, the frigate spoke some American whalers, from whom we learned that they knew nothing about the narwhal. But one of them, the captain of the Monroe, knowing that Ned Land had shipped on board the Abraham Lincoln, begged for his help in chasing a whale they had in sight.")

(defmethod annotator-key->execute-operation :lemma [k ann]
  (annotation->token-based-results ann LemmaResult))

;; :ner
(def ner-paragraph "Joe was the last person to see Fred. He saw him in Boston at McKenzie's pub at 3:00 where he paid $2.45 for an ale. Joe wanted to go to Vermont for the day to visit a cousin who works at IBM, but Sally and he had to look for Fred.")

(defmethod annotator-key->execute-operation :ner [k ann]
  (annotation->token-based-results ann NerResult))

;;;
(defrecord PerOperationResult [operation result])

;;;
;;; Main function
;;;
(defn analyse-text [text & annotators-keys]
  (let [annotation (Annotation. text)
        pipeline (apply make-pipeline annotators-keys)]
    (.annotate pipeline annotation) ;; side effect
    (mapv #(->PerOperationResult %
                                 (annotator-key->execute-operation % annotation))
          annotators-keys)))

(defonce paragraph "Let's pause, and then reflect.")

;; (.prettyPrint pipeline annotation *out*)

;; Sentence #1 (4 tokens):
;; Who are you?

;; Tokens:
;; [Text=Who CharacterOffsetBegin=0 CharacterOffsetEnd=3 PartOfSpeech=WP]
;; [Text=are CharacterOffsetBegin=4 CharacterOffsetEnd=7 PartOfSpeech=VBP]
;; [Text=you CharacterOffsetBegin=8 CharacterOffsetEnd=11 PartOfSpeech=PRP]
;; [Text=? CharacterOffsetBegin=11 CharacterOffsetEnd=12 PartOfSpeech=.]

;; Constituency parse:
;; (ROOT
;;  (SBARQ
;;   (WHNP (WP Who))
;;   (SQ (VBP are)
;;       (NP (PRP you)))
;;   (. ?)))


;; Dependency Parse (enhanced plus plus dependencies):
;; root(ROOT-0, Who-1)
;; cop(Who-1, are-2)
;; nsubj(Who-1, you-3)
;; punct(Who-1, ?-4)
;; nil

