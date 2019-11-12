# nlp.core

Clojure wrapper library of Stanford CoreNLP.

Internally it uses Annotation Pipeline and currently supports:

* :tokenize
* :pos
* :parse
* :lemma
* :ner
* :sentiment
* :dcoref
* :coref

## Usage

```
(analyse-text "The basic principles of contract law are discussed in this chapter. These apply to purely commercial transactions (such as between a manufacturing business and its supplier), as well as transactions where one of the parties is a consumer (a consumer is a person who acquires goods or services for personal or household use)."
              :sentiment :parse)
```

## License

Copyright © 2019 Jong-won Choi<oz.jongwon.choi@gmail.com>

Distributed under the MIT License. See LICENSE for details.
