#ifndef CSVITEMMODEL_H
#define CSVITEMMODEL_H

#include <QAbstractItemModel>
#include <mlpack/core/arma_extend/arma_extend.hpp>
#include <mlpack/core.hpp>

class CSVItemModel : public QAbstractItemModel
{
		Q_OBJECT

	public:
		explicit CSVItemModel(QObject *parent = nullptr, QString fileName = "german.csv");

		// Header:
		QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

		// Basic functionality:
		QModelIndex index(int row, int column,
						  const QModelIndex &parent = QModelIndex()) const override;
		QModelIndex parent(const QModelIndex &index) const override;

		int rowCount(const QModelIndex &parent = QModelIndex()) const override;
		int columnCount(const QModelIndex &parent = QModelIndex()) const override;

		QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;

	private:
		arma::mat csvData;
};

#endif // CSVITEMMODEL_H
