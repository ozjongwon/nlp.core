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
;;; * Classifier
;;; https://nlp.stanford.edu/wiki/Software/Classifier
;;;
;;;-----------------------------------------------------------------
;;; * Design & Implementation
;;;
;;; - Operation based, and operations are 'Annotations'
;;; - Operations have different levels to perform operation:
;;;   -- Token level - :tokenize, :pos, :lemma, :ner
;;;   -- Sentence level - :sentiment
;;;
(ns nlp.core
  (:gen-class)
  (:require
   [clojure.string :refer [join]]
   [clojure.set :refer [intersection]]
   [nlp.utils :refer [find-in-coll make-keyword atom?]]
   [nlp.records :refer [make-token-based-operation-result result-class->prototype
                        token-ann->token-map prototype->token-annotation-class
                        operation-keys->result-record prototype->make-token-based-operation-result
                        key->property-dependency annotators-keys->op-dispatch-set]]
   [medley.core :refer [find-first
                        ;;assoc-some
                        ]])
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

   [edu.stanford.nlp.sentiment SentimentCoreAnnotations$SentimentAnnotatedTree]
   [edu.stanford.nlp.neural.rnn RNNCoreAnnotations]

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
                                              (.put "annotators" (join \, annotators-opts))
                                              ;; A custom model example
                                              ;; jar tf stanford-corenlp-3.9.2-models.jar
                                              ;; edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger
                                              ;; (.put "pos.model"  "edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger")
                                              )
                                            true)]
         (reset! existing-core-nlp new-core-nlp)
         (reset! existing-opts-set annotators-opts-set)
         new-core-nlp)))))

;; :parse

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
#_
(defn- tree->parse-tree [tree-node]
  (convert-tree (read-string (.toString tree-node)) make-keyword name))

#_
(defn- tree->pos [tree-node]
  (->> (.taggedLabeledYield tree-node)
       (mapv #(vector (.word %) (make-keyword (.tag %))))
       (merge (hash-map))))
#_
(defn- sentence-ann->token-based-result [result-class subkeys sentence-ann]
  (->>
       (.get sentence-ann)
       (mapv #(make-token-based-operation-result result-class subkeys %))))

(defrecord SentenceResult [tokens sentences])

(defmulti execute-sentence-based-operation (fn [op-key & _] op-key))

(defmethod execute-sentence-based-operation :sentiment [_ sentence-ann]
  ;; Sentiment score
  ;;     Very negative = 0
  ;;     Negative = 1
  ;;     Neutral = 2
  ;;     Positive = 3
  ;;     Very positive = 4
  (. RNNCoreAnnotations getPredictedClass (.get sentence-ann SentimentCoreAnnotations$SentimentAnnotatedTree)))

(defn execute-sentence-based-operations [sentence-ann sentence-operation-set]
  ;; assume there is no operations dependency
  (let [prototype (-> (mapv :key sentence-operation-set)
                      (operation-keys->result-record)
                      (result-class->prototype))
        #_ {}]
    (reduce (fn [result op-key]
              (assoc result op-key (execute-sentence-based-operation op-key sentence-ann)))
            prototype
            #_ (mapv :key sentence-operation-set)
            (keys prototype))))

(defn- execute-annotation-operations [ann {:keys [token sentence]}]
  (let [token-based-result-class (operation-keys->result-record (mapv :key token))
        token-prototype (result-class->prototype token-based-result-class)
        tokens-ann-class (prototype->token-annotation-class token-prototype)]
    (mapv (fn [sentence-ann]
            (->SentenceResult (mapv #(prototype->make-token-based-operation-result token-prototype %)
                                    (.get sentence-ann tokens-ann-class))
                              (execute-sentence-based-operations sentence-ann sentence)))
          (.get ann CoreAnnotations$SentencesAnnotation))))

;; :lemma
(def lemma-paragraph "Similar to stemming is Lemmatization. This is the process of finding its lemma, its form as found in a dictionary.")

(def pos-paragraph "Bill used the force to force the manger to tear the bill in two.")
(def pos-paragraph2 "AFAIK she H8 cth! BTW had a GR8 tym at the party BBIAM.")
(def pos-paragraph3 "Whether \"Blue\" was correct or not (it's not) is debatable.")

(def pos-paragraph4 "The voyage of the Abraham Lincoln was for a long time marked by no special incident. But one circumstance happened which showed the wonderful dexterity of Ned Land, and proved what confidence we might place in him. The 30th of June, the frigate spoke some American whalers, from whom we learned that they knew nothing about the narwhal. But one of them, the captain of the Monroe, knowing that Ned Land had shipped on board the Abraham Lincoln, begged for his help in chasing a whale they had in sight.")

;; :ner
(def ner-paragraph "Joe was the last person to see Fred. He saw him in Boston at McKenzie's pub at 3:00 where he paid $2.45 for an ale. Joe wanted to go to Vermont for the day to visit a cousin who works at IBM, but Sally and he had to look for Fred.")


;;;FIXME: YOU'RE HERE
;;; Sentence based
;;;
#_
(defn- annotation->sentence-based-results [ann result-class]
  (->> (.get ann CoreAnnotations$SentencesAnnotation)
       (mapv #(sentence-ann->sentence-based-result % result-class))))

;;;
;;; Main function
;;;
(defn analyse-text [text & annotators-keys]
  (let [annotation (Annotation. text)
        pipeline (apply make-pipeline annotators-keys)]
    (.annotate pipeline annotation) ;; side effect
    (execute-annotation-operations annotation (annotators-keys->op-dispatch-set annotators-keys))
    #_
    (->AnalyseResult
                     1
                     #_
                     (execute-sentence-based-operations  annotation))))

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

