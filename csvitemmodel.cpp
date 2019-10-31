#include "csvitemmodel.h"

#include <mlpack/core.hpp>
#include <mlpack/core/cv/cv_base.hpp>
#include <mlpack/core/cv/metrics/accuracy.hpp>
#include <mlpack/core/cv/metrics/precision.hpp>
#include <mlpack/core/cv/k_fold_cv.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/core/arma_extend/arma_extend.hpp>


using namespace mlpack;
using namespace mlpack::data;
using namespace arma;

CSVItemModel::CSVItemModel(QObject *parent, QString fileName)
	: QAbstractItemModel(parent)
{
	if(!Load(fileName.toStdString(), csvData))
	{
		csvData.zeros();
	}
}

QVariant CSVItemModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	// FIXME: Implement me!
	return QVariant();
}

QModelIndex CSVItemModel::index(int row, int column, const QModelIndex &parent) const
{
	// FIXME: Implement me!
	return QModelIndex();
}

QModelIndex CSVItemModel::parent(const QModelIndex &index) const
{
	// FIXME: Implement me!
	return QModelIndex();
}

int CSVItemModel::rowCount(const QModelIndex &parent) const
{
	if (!parent.isValid())
		return 0;

	// FIXME: Implement me!
	return 0;
}

int CSVItemModel::columnCount(const QModelIndex &parent) const
{
	if (!parent.isValid())
		return 0;

	// FIXME: Implement me!
	return 0;
}

QVariant CSVItemModel::data(const QModelIndex &index, int role) const
{
	if (!index.isValid())
		return QVariant();

	// FIXME: Implement me!
	return QVariant();
}
