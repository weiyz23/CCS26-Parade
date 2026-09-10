#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <stdexcept>

extern "C" {
#include <bgpstream.h>
}

class BGPStreamParser {
public:
    // Filter type enumeration
    enum class FilterType {
        PROJECT,
        COLLECTOR,
        ROUTER,
        PEER_ASN,
        ORIGIN_ASN,
        PREFIX,
        COMMUNITY,
        ASPATH,
        RECORD_TYPE,
        RESOURCE_TYPE
    };

    // Data interface type
    enum class DataInterface {
        SINGLEFILE,
        BROKER,
        SQLITE
    };

private:
    std::unique_ptr<bgpstream_t, decltype(&bgpstream_destroy)> stream_;
    std::unordered_map<std::string, std::string> interface_options_;
    std::vector<std::pair<FilterType, std::string>> filters_;
    DataInterface data_interface_ = DataInterface::SINGLEFILE;
    uint32_t interval_start_ = 0;
    uint32_t interval_end_ = BGPSTREAM_FOREVER;
    bool live_mode_ = false;
    int record_limit_ = -1;

    static bgpstream_filter_type_t convert_filter_type(FilterType type) {
        switch (type) {
            case FilterType::PROJECT: return BGPSTREAM_FILTER_TYPE_PROJECT;
            case FilterType::COLLECTOR: return BGPSTREAM_FILTER_TYPE_COLLECTOR;
            case FilterType::ROUTER: return BGPSTREAM_FILTER_TYPE_ROUTER;
            case FilterType::PEER_ASN: return BGPSTREAM_FILTER_TYPE_ELEM_PEER_ASN;
            case FilterType::ORIGIN_ASN: return BGPSTREAM_FILTER_TYPE_ELEM_ORIGIN_ASN;
            case FilterType::PREFIX: return BGPSTREAM_FILTER_TYPE_ELEM_PREFIX;
            case FilterType::COMMUNITY: return BGPSTREAM_FILTER_TYPE_ELEM_COMMUNITY;
            case FilterType::ASPATH: return BGPSTREAM_FILTER_TYPE_ELEM_ASPATH;
            case FilterType::RECORD_TYPE: return BGPSTREAM_FILTER_TYPE_RECORD_TYPE;
            case FilterType::RESOURCE_TYPE: return BGPSTREAM_FILTER_TYPE_RESOURCE_TYPE;
            default: throw std::invalid_argument("Invalid filter type");
        }
    }

    static bgpstream_data_interface_id_t convert_data_interface(DataInterface di) {
        switch (di) {
            case DataInterface::SINGLEFILE: return BGPSTREAM_DATA_INTERFACE_SINGLEFILE;
            case DataInterface::BROKER: return BGPSTREAM_DATA_INTERFACE_BROKER;
            case DataInterface::SQLITE: return BGPSTREAM_DATA_INTERFACE_SQLITE;
            default: throw std::invalid_argument("Invalid data interface");
        }
    }

public:
    BGPStreamParser() : stream_(bgpstream_create(), bgpstream_destroy) {
        if (!stream_) {
            throw std::runtime_error("Failed to create BGPStream");
        }
    }

    // Streaming configuration interface
    BGPStreamParser& set_data_interface(DataInterface di) {
        data_interface_ = di;
        return *this;
    }

    BGPStreamParser& add_filter(FilterType type, const std::string& value) {
        filters_.emplace_back(type, value);
        return *this;
    }

    BGPStreamParser& add_interface_option(const std::string& key, const std::string& value) {
        interface_options_[key] = value;
        return *this;
    }

    BGPStreamParser& set_rib_file(const std::string& filepath) {
        return set_data_interface(DataInterface::SINGLEFILE)
               .add_interface_option("rib-file", filepath);
    }

    BGPStreamParser& set_update_file(const std::string& filepath) {
        return set_data_interface(DataInterface::SINGLEFILE)
               .add_interface_option("upd-file", filepath);
    }

    BGPStreamParser& set_time_interval(uint32_t start, uint32_t end = BGPSTREAM_FOREVER) {
        interval_start_ = start;
        interval_end_ = end;
        return *this;
    }

    BGPStreamParser& set_live_mode(bool live = true) {
        live_mode_ = live;
        return *this;
    }

    BGPStreamParser& set_record_limit(int limit) {
        record_limit_ = limit;
        return *this;
    }

    // Initialize BGPStream
    bool initialize() {
        try {
            // Set data interface
            bgpstream_set_data_interface(stream_.get(), convert_data_interface(data_interface_));

            // Add filter
            for (const auto& [type, value] : filters_) {
                if (!bgpstream_add_filter(stream_.get(), convert_filter_type(type), value.c_str())) {
                    throw std::runtime_error("Failed to add filter: " + value);
                }
            }

            // Set interface options
            for (const auto& [key, value] : interface_options_) {
                auto option = bgpstream_get_data_interface_option_by_name(
                    stream_.get(), convert_data_interface(data_interface_), key.c_str());
                if (!option) {
                    throw std::runtime_error("Invalid option: " + key);
                }
                if (bgpstream_set_data_interface_option(stream_.get(), option, value.c_str()) != 0) {
                    throw std::runtime_error("Failed to set option: " + key + "=" + value);
                }
            }

            // Set time interval
            if (interval_start_ != 0) {
                if (!bgpstream_add_interval_filter(stream_.get(), interval_start_, interval_end_)) {
                    throw std::runtime_error("Failed to set time interval");
                }
            }

            // Set live mode
            if (live_mode_) {
                bgpstream_set_live_mode(stream_.get());
            }

            // Start stream
            if (bgpstream_start(stream_.get()) < 0) {
                throw std::runtime_error("Failed to start BGPStream");
            }

            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "BGPStream initialization error: %s\n", e.what());
            return false;
        }
    }

    // Callback interface for parsing records
    template<typename RecordCallback>
    int parse_records(RecordCallback&& callback) {
        bgpstream_record_t* record = nullptr;
        int count = 0;

        while (bgpstream_get_next_record(stream_.get(), &record) > 0) {
            if (record->status != BGPSTREAM_RECORD_STATUS_VALID_RECORD) {
                continue;
            }

            if (record_limit_ > 0 && count >= record_limit_) {
                break;
            }

            if (!callback(record)) {
                break;
            }

            count++;
        }

        return count;
    }

    // Get statistics
    void print_info() const {
        printf("BGPStream Parser Configuration:\n");
        printf("  Data Interface: %d\n", static_cast<int>(data_interface_));
        printf("  Filters: %zu\n", filters_.size());
        printf("  Interface Options: %zu\n", interface_options_.size());
        printf("  Time Interval: %u - %u\n", interval_start_, interval_end_);
        printf("  Live Mode: %s\n", live_mode_ ? "yes" : "no");
        printf("  Record Limit: %d\n", record_limit_);
    }
};

// Convenience function, Builder class rewritten with C++ logic
class BGPStreamParserBuilder {
private:
    BGPStreamParser::DataInterface data_interface_ = BGPStreamParser::DataInterface::SINGLEFILE;
    std::unordered_map<std::string, std::string> interface_options_;
    std::vector<std::pair<BGPStreamParser::FilterType, std::string>> filters_;
    uint32_t interval_start_ = 0;
    uint32_t interval_end_ = BGPSTREAM_FOREVER;
    bool live_mode_ = false;
    int record_limit_ = -1;

public:
    BGPStreamParserBuilder& set_data_interface(BGPStreamParser::DataInterface di) {
        data_interface_ = di;
        return *this;
    }

    BGPStreamParserBuilder& add_filter(BGPStreamParser::FilterType type, const std::string& value) {
        filters_.emplace_back(type, value);
        return *this;
    }

    BGPStreamParserBuilder& add_interface_option(const std::string& key, const std::string& value) {
        interface_options_[key] = value;
        return *this;
    }

    BGPStreamParserBuilder& set_rib_file(const std::string& filepath) {
        return set_data_interface(BGPStreamParser::DataInterface::SINGLEFILE)
               .add_interface_option("rib-file", filepath);
    }

    BGPStreamParserBuilder& set_update_file(const std::string& filepath) {
        return set_data_interface(BGPStreamParser::DataInterface::SINGLEFILE)
               .add_interface_option("upd-file", filepath);
    }

    BGPStreamParserBuilder& set_time_interval(uint32_t start, uint32_t end = BGPSTREAM_FOREVER) {
        interval_start_ = start;
        interval_end_ = end;
        return *this;
    }

    BGPStreamParserBuilder& set_live_mode(bool live = true) {
        live_mode_ = live;
        return *this;
    }

    BGPStreamParserBuilder& set_record_limit(int limit) {
        record_limit_ = limit;
        return *this;
    }

    // Build and return configured parser
    std::unique_ptr<BGPStreamParser> build() {
        auto parser = std::make_unique<BGPStreamParser>();
        
        // Apply all configurations
        parser->set_data_interface(data_interface_);
        
        for (const auto& [type, value] : filters_) {
            parser->add_filter(type, value);
        }
        
        for (const auto& [key, value] : interface_options_) {
            parser->add_interface_option(key, value);
        }
        
        if (interval_start_ != 0) {
            parser->set_time_interval(interval_start_, interval_end_);
        }
        
        if (live_mode_) {
            parser->set_live_mode(live_mode_);
        }
        
        if (record_limit_ > 0) {
            parser->set_record_limit(record_limit_);
        }
        
        return parser;
    }

    // Convenient static methods
    static BGPStreamParserBuilder for_rib_file(const std::string& filepath) {
        return BGPStreamParserBuilder().set_rib_file(filepath);
    }

    static BGPStreamParserBuilder for_update_file(const std::string& filepath) {
        return BGPStreamParserBuilder().set_update_file(filepath);
    }

    static BGPStreamParserBuilder for_collector(const std::string& collector_name) {
        return BGPStreamParserBuilder()
            .set_data_interface(BGPStreamParser::DataInterface::BROKER)
            .add_filter(BGPStreamParser::FilterType::COLLECTOR, collector_name);
    }
};