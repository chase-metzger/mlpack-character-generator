#include "mainwindow.h"

#include <QApplication>
#include <QFile>
#include <QStringList>
#include <QVector>
#include <QDebug>

#include <mlpack/core/cv/cv_base.hpp>
#include <mlpack/core/cv/metrics/accuracy.hpp>
#include <mlpack/core/cv/metrics/precision.hpp>
#include <mlpack/core/cv/metrics/mse.hpp>
#include <mlpack/core/cv/k_fold_cv.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/lstm.hpp>
#include <mlpack/methods/ann/layer/dropout.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/core/arma_extend/arma_extend.hpp>

template<typename T = size_t, size_t... Is>
auto make_range_impl(const T &max, std::index_sequence<Is...>)
{
	//static_assert((sizeof...(Is) == max - 1), "make_range_impl: Bad Is...");
	return std::make_tuple(Is...);
}

template<typename T, T Max, typename Indices = std::make_index_sequence<Max>>
auto make_range(const T &max = Max)
{
	return make_range_impl<T>(max, Indices{});
}

template<typename Func, size_t... Is>
auto for_range_impl(Func &&func, std::index_sequence<Is...>)
{
//	return std::invoke(std::forward<Func>(func), std::initializer_list<size_t>{Is...});
	return std::initializer_list<size_t>{Is...};
}

template<typename T, T Max, typename Func, typename Indices = std::make_index_sequence<Max>>
auto for_range(Func &&func)
{
//	std::invoke(std::forward<Func>(func), Indices{});
	return for_range_impl(std::forward<Func>(func), Indices{});
}

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	MainWindow w;

	using namespace mlpack;
	using namespace mlpack::data;
	using namespace mlpack::tree;
	using namespace mlpack::cv;
	using namespace mlpack::ann;
	using namespace arma;

	QFile file("sentiments.csv");
	if(!file.open(QIODevice::ReadOnly))
	{
		return -1;
	}

	QList<QStringList> lines;
	QStringList header;
	int count = 0;
	//PARSE the file
	while(!file.atEnd() && count <= 500)
	{
		auto buffer = file.readLine();
		auto splitBuffer = buffer.split(',');
		QStringList parsedLine;
		for(const auto &elem : splitBuffer)
		{
			parsedLine.append(elem);
		}
		lines.append(parsedLine);
		++count;
	}

	const int hiddenSize = 128;
	const int numLetters = 256;

	//GRAB header from first line
	header = lines[0];
	lines.removeAt(0);
	qDebug() << "Headers for sentiments.csv: " << header << "\n";

	//REMOVE ItemID
	for(auto &line : lines)
	{
		line.removeAt(0);
	}

	std::cout << lines[0].first().toStdString() << std::endl;

	std:vector<double> sentiments;
	std::vector<double> sentimentTexts;
	sentiments.reserve(lines.size());
	int maxLengthLineCount = 0;
	for(const auto &elem : lines)
	{
		int count = elem[1].size();
		if(count > maxLengthLineCount)
			maxLengthLineCount = count;
	}

	auto vectorLines = lines.toVector().toStdVector();
	const int MAX_LINE_LENGTH = 512;
	for(auto &line : lines)
	{
		if(line.size() < MAX_LINE_LENGTH)
		{
			std::vector<QString> padding(std::abs(line.size() - MAX_LINE_LENGTH));
			std::fill(padding.begin(), padding.end(), "<PAD>");
			QStringList paddingList(QList<QString>::fromVector(QVector<QString>::fromStdVector(padding)));
			line.append(paddingList);
		}
	}

	QString fullText;
	fullText.reserve(10000);
	for(auto &line : lines)
	{
		line.removeAt(0);
		for(const auto &ch : line)
		{
			fullText += ch;
		}
	}


	for(const auto &ch : fullText)
	{
		sentimentTexts.push_back(static_cast<double>(ch.toLatin1()));
	}

	using MatType = cube;

	MatType datasetLabels(sentiments.data(), sentiments.size(), 1, 1);
	MatType datasetTexts(sentimentTexts.data(), sentimentTexts.size(), 1, 1);

	MatType trainData = datasetTexts.subcube(0, 0, 0, static_cast<uword>(datasetTexts.n_rows * 0.8), datasetTexts.n_cols - 1, 0);

	MatType testData = datasetTexts.subcube(static_cast<uword>(datasetTexts.n_rows * 0.8), 0, 0, datasetTexts.n_rows - 1, datasetTexts.n_cols - 1, 0);


	FFN<NegativeLogLikelihood<cube, cube>, RandomInitialization> net;
	RNN<> rnn(1);
	net.Add<LSTM<>>(numLetters, hiddenSize, 2);
//	net.Add<Dropout<>>(0.1);
	net.Add<Linear<>>(hiddenSize, numLetters);

	rnn.Add<LSTM<>>(numLetters, hiddenSize, 1);
//	rnn.Add<Dropout<>>(0.1);
	rnn.Add<Linear<>>(numLetters, hiddenSize);


	const auto makeInput = [](const char *line) -> MatType {
		const auto strLen = strlen(line);
		cube result(strLen, 1, numLetters, fill::zeros);
		for(int i = 0; i < strLen; ++i)
		{
			const auto letter = line[i];
			result.at(i, 0, static_cast<uword>(letter)) = 1.0;
//			result.at(static_cast<uword>(letter + 1), 0, i) = 1.0;
//			result.at(static_cast<uword>(letter + 1), 0, i) = 1.0;
		}

		return result;
	};

	const auto makeTarget = [] (const char *line) -> MatType {
		const auto strLen = strlen(line);
//		std::vector<double> letterIndices(strLen);
		MatType result(1, 1, strLen, fill::zeros);
//		MatType result(1, 1, strLen, fill::zeros);
		for(int i = 0; i < strLen; ++i)
		{
			const auto letter = line[i];
			const auto letterIndex = static_cast<double>(letter + 1);
			result.at(0, 0, i) = letterIndex;
//			result.at(i, 0, 0) = letterIndex;
//			result.at(i, 0, static_cast<uword>(letter)) = static_cast<double>(letter);
//			result.at(0, i, static_cast<uword>(letter)) = static_cast<double>(letter);
//			result.at(static_cast<uword>(letter), i, 0) = static_cast<double>(letter);
		}
		return result;
//		return cube(letterIndices.data(), MAX_LINE_LENGTH, 1, 1);
	};

	std::vector<std::string> trainDataLines;
	for(int i = 0; i < trainData.n_cols; ++i)
	{
		std::string line;
		for(int j = 0; j < trainData.col(i).n_cols && j < MAX_LINE_LENGTH; ++j)
		{
			const auto elem = static_cast<char>(trainData.col(i).at(j, i, 0));
			line += elem;
		}
		trainDataLines.push_back(line);
	}

	std::vector<cube> inputs(trainDataLines.size());
	std::vector<cube> targets(trainDataLines.size());
	std::map<std::string, std::vector<std::string>> categories;
	MatType inputCategory(1, 1, 1);
	inputCategory = inputCategory.zeros();
	inputCategory.at(0, 0, 0) = 1.0;
	for(int i = 0; i < trainDataLines.size(); ++i)
	{
		inputs[i] = makeInput(trainDataLines[i].c_str());
		targets[i] = makeTarget(trainDataLines[i].c_str());
	}
	std::cout << inputs[0].slice(0) << std::endl;
	std::cout << targets[0][22] << std::endl;
	categories["ENG"] = trainDataLines;

	const auto tempInput = makeInput("A");
	const auto tempTarget = makeTarget("A");

	std::cout << tempInput << "\n" << tempTarget << std::endl;

#if 1
	const int numEpochs = 2;
	for(int i = 0; i < numEpochs; ++i)
	{
		net.Train(inputs[i], targets[i]);
	//	rnn.Train(inputs[i], targets[i]);
	}
#endif

//	net.Train(trainData, trainLabels);

//	mat assignment;
//	net.Predict(testData, assignment);

//	std::cout << "Predictions: " << assignment << "\nCorrect responses: " << testLabels;

//	net.Add<Linear<>>(trainData.n_rows, 8);
//	net.Add<Linear<>>(trainData.n_rows, 8);
//	net.Add<Linear<>>(trainData.n_rows, 8);
//	net.Add<Linear<>>(trainData.n_rows, 8);

//	{
//		return -1;
//	}

//	Row<size_t> labels;
//	labels = conv_to<Row<size_t>>::from(dataset.row(dataset.n_rows - 1));
//	dataset.shed_row(dataset.n_rows - 1);

//	std::cout << labels << std::endl;


//	std::cout << dataset.n_elem << std::endl;
//	std::stringstream ss;
//	ss << dataset << std::endl;

	//w.setDataset(trainData);
	w.setResultText("Result: unknown");

	//FFN<> net;
//	net.Add<

	w.show();
	return a.exec();
}


//#include "mainwindow.h"

//#include <QApplication>

//#include <mlpack/core.hpp>
//#include <mlpack/core/cv/cv_base.hpp>
//#include <mlpack/core/cv/metrics/accuracy.hpp>
//#include <mlpack/core/cv/metrics/precision.hpp>
//#include <mlpack/core/cv/k_fold_cv.hpp>
//#include <mlpack/methods/ann/rnn.hpp>
//#include <mlpack/methods/ann/ffn.hpp>
//#include <mlpack/methods/random_forest/random_forest.hpp>
//#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
//#include <mlpack/core/arma_extend/arma_extend.hpp>

//int main(int argc, char *argv[])
//{
//	QApplication a(argc, argv);
//	MainWindow w;

//	using namespace mlpack;
//	using namespace mlpack::data;
//	using namespace mlpack::tree;
//	using namespace mlpack::cv;
//	using namespace mlpack::ann;
//	using namespace arma;

////	FFN<> net;
////	net.Add<
//	mat testM;

//	if(!Load("german.csv", testM, false))
//	{
//		std::cout << "Failed to load data" << std::endl;
//		return -1;
//	}

//	Row<size_t> labels;

//	labels = conv_to<Row<size_t>>::from(testM.row(testM.n_rows - 1));
//	testM.shed_row(testM.n_rows - 1);

//	const size_t NUM_CLASSES = 2; //POS AND NEG
//	const size_t MIN_LEAF_SIZE = 5;
//	const size_t NUM_TREES = 10;
//	const size_t k = 10;

//	RandomForest<GiniGain, RandomDimensionSelect> rf;
//	rf = RandomForest<GiniGain, RandomDimensionSelect>(testM, labels, NUM_CLASSES, NUM_TREES, MIN_LEAF_SIZE);

//	Row<size_t> predictions;
//	rf.Classify(testM, predictions);
//	const size_t correct = accu(predictions == labels);
//	cout << "\n Training Accuracy: " << (double(correct) / double(labels.n_elem));

//	double precision = Precision<Binary>::Evaluate(rf, testM, labels);
//	std::cout << "\n Precision: " << precision;

//	double accuracy = Accuracy::Evaluate(rf, testM, labels);
//	std::cout << "\n Accuracy: " << accuracy;

//	Save("german_model.xml", "model", rf, false);

//	Load("german_model.xml", "model", rf);

//	mat sample("2 12 2 13 1 2 2 1 3 24 3 1 1 1 1 1 0 1 0 1 0 0 0");

//	std::cout << "\nClassifying new sample: " << sample << std::endl;

//	mat probabilities;

//	rf.Classify(sample, predictions, probabilities);

//	u64 result = predictions.at(0);

//	std::stringstream ss;
//	ss << "\nClassification result: " << result << ", Probabilities: " << probabilities.at(0) << "/" << probabilities.at(1) << std::endl;

//	QString resultText = ss.str().c_str();

//	w.setResultText(resultText);

//	w.show();
//	return a.exec();
//}
